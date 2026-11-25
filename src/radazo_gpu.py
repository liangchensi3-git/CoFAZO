"""Device-compatible wrapper; optimization math stays aligned with radazo.py."""

import logging

import torch

from radazo_official import R_AdaZO


logger = logging.getLogger(__name__)


class R_AdaZO_Device(R_AdaZO):
    """Use a device-local generator without changing the official update math."""

    def __init__(self, *args, device=None, moment_dtype=None, **kwargs):
        self.device = device if device is not None else torch.device("cpu")
        self.moment_dtype = moment_dtype
        super().__init__(*args, **kwargs)

    def _get_perturbation(self, shape, seed):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        # Keep the official float32 direction and unit-L2 normalization.
        u = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        u_norm = torch.norm(u)
        if u_norm > 0:
            u = u / u_norm
        return u

    def _check_finite(self, label, value, param_index, seed=None):
        if not bool(torch.isfinite(value).all()):
            name = self.parameter_names[param_index] if hasattr(self, "parameter_names") else str(param_index)
            raise FloatingPointError(
                f"Official R-AdaZO non-finite {label}: step={self.state['step']}, "
                f"parameter={name}, index={param_index}, seed={seed}, "
                f"abs_max={float(value.detach().float().abs().max())}"
            )

    @torch.no_grad()
    def step(self, closure):
        # Official step order and arithmetic, plus fail-fast finite diagnostics.
        if closure is None:
            raise ValueError("RaZO requires closure to evaluate function value")
        self.state['step'] += 1
        for group in self.param_groups:
            for param_index, p in enumerate(group['params']):
                state = self.state[p]
                if len(state) == 0:
                    state_dtype = self.moment_dtype or p.dtype
                    state['exp_avg'] = torch.zeros_like(p, dtype=state_dtype, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=state_dtype, memory_format=torch.preserve_format)
                mu = group['mu']
                n_samples = group['n_samples']
                beta1, beta2 = group['betas']
                grad_est = torch.zeros_like(p)
                for seed in self.state['seeds']:
                    current_grad, loss = self._estimate_gradient(p, closure, mu, seed)
                    self._check_finite("loss", loss, param_index, seed)
                    self._check_finite("current_grad", current_grad, param_index, seed)
                    grad_est.add_(current_grad)
                grad_est.div_(n_samples)
                self._check_finite("grad_est", grad_est, param_index)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad_est, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(exp_avg, exp_avg, value=(1 - beta2))
                self._check_finite("exp_avg", exp_avg, param_index)
                self._check_finite("exp_avg_sq", exp_avg_sq, param_index)

                step_size = group['lr']
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                self._check_finite("parameter_after_update", p.data, param_index)
                if self.state['step'] == 1 and param_index < 5:
                    logger.info(
                        "OFFICIAL_DIAG step=1 index=%d loss=%.9e grad_rms=%.6e "
                        "m_absmax=%.6e v_absmax=%.6e param_absmax=%.6e",
                        param_index, float(loss), float(grad_est.float().square().mean().sqrt()),
                        float(exp_avg.abs().max()), float(exp_avg_sq.abs().max()),
                        float(p.data.abs().max()),
                    )
        return loss
