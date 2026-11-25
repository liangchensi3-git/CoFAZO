"""Memory-efficient adaptive ZO updates for MeZO and FZOO gradient estimates.

Implements ZO-AdaMM, ZO-RMSProp, and R-AdaZO (ICML 2025,
https://github.com/shuyao95/R-AdaZO, official commit f7c9377), plus their
Rademacher/N-direction FZOO counterparts. R-AdaZO's only algorithmic difference
from ZO-AdaMM is that v accumulates m^2 instead of g^2. Moment buffers are fp32
and no bias correction is applied by default, matching the paper and official
optimizer. Only trainable parameters need moment buffers.
"""

import torch


ADAPTIVE_ZO_METHODS = ("ZO-AdaMM", "ZO-RMSProp", "R-AdaZO")
FZOO_ADAPTIVE_METHODS = ("FZOO-AdaMM", "FZOO-R-AdaZO", "FZOO-Decoupled")
ALL_ADAPTIVE_ZO_METHODS = ADAPTIVE_ZO_METHODS + FZOO_ADAPTIVE_METHODS
ZO_DIRECTIONS = ("rademacher", "gaussian")
ZO_SECOND_MOMENTS = ("m2", "g2")


def fzoo_uses_m2(method, step, g2_warmup=0, g2_after=0):
    """Select the second-moment input at a 1-based optimizer step."""
    if step < 1 or g2_warmup < 0 or g2_after < 0:
        raise ValueError("Invalid adaptive FZOO step or switch boundary")
    if g2_after and g2_after <= g2_warmup:
        raise ValueError("g2_after must be zero or greater than g2_warmup")
    if method == "FZOO-AdaMM":
        return False
    if method != "FZOO-R-AdaZO":
        raise ValueError(f"Second-moment switching is not defined for {method}")
    if g2_after and step > g2_after:
        return False
    return step > g2_warmup


def sample_zo_direction(param, direction="rademacher", generator=None):
    """Draw a perturbation with the same dtype/device stream as the trainers."""
    if direction == "rademacher":
        return torch.randint(
            0,
            2,
            size=param.data.size(),
            device=param.data.device,
            dtype=param.data.dtype,
            generator=generator,
        ) * 2 - 1
    if direction == "gaussian":
        return torch.normal(
            mean=0,
            std=1,
            size=param.data.size(),
            device=param.data.device,
            dtype=param.data.dtype,
            generator=generator,
        )
    raise ValueError(f"Unknown ZO direction distribution: {direction}")


def adaptive_zo_betas(method, beta1, beta2):
    if method not in ALL_ADAPTIVE_ZO_METHODS:
        raise ValueError(f"Unknown adaptive ZO method: {method}")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("ZO moment coefficients must be in [0, 1)")
    if method == "ZO-RMSProp":
        beta1 = 0.0
    return beta1, beta2


@torch.no_grad()
def adaptive_zo_update(
    named_parameters,
    zo_random_seed,
    projected_grad,
    learning_rate,
    beta1,
    beta2,
    eps,
    state,
    refined_second_moment,
    diagnostics=None,
):
    """Apply one adaptive ZO step and return the (m, v) parameter-state list.

    Replays the same per-parameter Gaussian draws as mezotrainer's perturbation
    pass, so the full gradient vector is never materialized.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if state is None:
        state = [
            (
                torch.zeros_like(param, dtype=torch.float32),
                torch.zeros_like(param, dtype=torch.float32),
            )
            for _, param in named_parameters
        ]
    if len(state) != len(named_parameters):
        raise ValueError("Adaptive ZO state does not match trainable parameters")

    if diagnostics is not None:
        diagnostics.update(
            numel=0, g_sq=0.0, m_sq=0.0, sqrt_v_sq=0.0,
            update_sq=0.0, update_max_abs=0.0,
            actual_delta_sq=0.0, actual_delta_max_abs=0.0,
        )

    torch.manual_seed(zo_random_seed)
    for (name, param), (m, v) in zip(named_parameters, state):
        z = sample_zo_direction(param, "gaussian")
        g = z.float() * projected_grad
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        second_moment_input = m if refined_second_moment else g
        v.mul_(beta2).addcmul_(
            second_moment_input, second_moment_input, value=1 - beta2
        )
        update = m / (v.sqrt() + eps)
        param.data.add_(update.to(param.data.dtype), alpha=-learning_rate)
        if diagnostics is not None:
            sqrt_v = v.sqrt()
            delta = update.to(param.data.dtype) * (-learning_rate)
            diagnostics["numel"] += param.numel()
            diagnostics["g_sq"] += float(torch.dot(g.ravel(), g.ravel()))
            diagnostics["m_sq"] += float(torch.dot(m.ravel(), m.ravel()))
            diagnostics["sqrt_v_sq"] += float(torch.dot(sqrt_v.ravel(), sqrt_v.ravel()))
            diagnostics["update_sq"] += float(torch.dot(update.ravel(), update.ravel()))
            diagnostics["update_max_abs"] = max(diagnostics["update_max_abs"], float(update.abs().max()))
            delta_float = delta.float()
            diagnostics["actual_delta_sq"] += float(torch.dot(delta_float.ravel(), delta_float.ravel()))
            diagnostics["actual_delta_max_abs"] = max(
                diagnostics["actual_delta_max_abs"], float(delta_float.abs().max())
            )
    return state


@torch.no_grad()
def adaptive_fzoo_update(
    named_parameters,
    zo_random_seeds,
    projected_grad,
    learning_rate,
    beta1,
    beta2,
    eps,
    state,
    refined_second_moment,
    direction="rademacher",
    bias_correction=False,
    step=None,
    diagnostics=None,
):
    """Apply R-AdaZO/ZO-AdaMM updates to FZOO's gradient estimate.

    The perturbation seeds are replayed per parameter. For each seed this draws
    exactly the same sequence of tensors as trainer.py's seed-major FZOO loop,
    while materializing only one temporary fp32 gradient tensor at a time.
    Bias correction and update-norm diagnostics are opt-in R13 instrumentation;
    the default path is numerically identical to R12.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if direction not in ZO_DIRECTIONS:
        raise ValueError(f"Unknown ZO direction distribution: {direction}")
    if len(zo_random_seeds) != len(projected_grad) or not zo_random_seeds:
        raise ValueError("FZOO seeds and projected gradients must be non-empty and aligned")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("FZOO moment coefficients must be in [0, 1)")
    if bias_correction and (step is None or step < 1):
        raise ValueError("Bias correction requires a 1-based optimizer step")
    if state is None:
        state = [
            (
                torch.zeros_like(param, dtype=torch.float32),
                torch.zeros_like(param, dtype=torch.float32),
            )
            for _, param in named_parameters
        ]
    if len(state) != len(named_parameters):
        raise ValueError("Adaptive FZOO state does not match trainable parameters")
    if not named_parameters:
        raise ValueError("Adaptive FZOO requires at least one trainable parameter")

    if diagnostics is not None:
        diagnostics.update(
            numel=0,
            g_sq=0.0,
            m_sq=0.0,
            m_raw_sq=0.0,
            sqrt_v_sq=0.0,
            update_sq=0.0,
            update_max_abs=0.0,
            actual_delta_sq=0.0,
            actual_delta_max_abs=0.0,
            probe_numel=0,
            probe_expected_delta_sq=0.0,
            probe_observed_delta_sq=0.0,
            probe_roundoff_max_abs=0.0,
            probe_expected_nonzero_observed_zero=0,
            m_prev_sq=0.0,
            m_prev_g_dot=0.0,
            m_prev_m_dot=0.0,
            sign_agree=0,
            sign_valid=0,
        )

    generators = [
        torch.Generator(device=named_parameters[0][1].data.device).manual_seed(int(seed))
        for seed in zo_random_seeds
    ]
    for (name, param), (m, v) in zip(named_parameters, state):
        g = torch.zeros_like(param, dtype=torch.float32)
        for generator, alpha in zip(generators, projected_grad):
            # Each generator advances across parameters in their original order,
            # reproducing trainer.py's seed-major stream without global reseeding
            # or materializing gradients for every parameter at once.
            z = sample_zo_direction(param, direction, generator)
            g.add_(z.float(), alpha=float(alpha))
        m_previous = m.clone() if diagnostics is not None else None
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        second_moment_input = m if refined_second_moment else g
        v.mul_(beta2).addcmul_(
            second_moment_input, second_moment_input, value=1 - beta2
        )
        if bias_correction:
            m_for_update = m / (1 - beta1 ** step)
            sqrt_v = (v / (1 - beta2 ** step)).sqrt()
        else:
            m_for_update = m
            sqrt_v = v.sqrt()
        update = m_for_update / (sqrt_v + eps)
        if diagnostics is None:
            param.data.add_(update.to(param.data.dtype), alpha=-learning_rate)
            continue

        delta = update.to(param.data.dtype) * (-learning_rate)
        probe_before = param.data.reshape(-1)[:min(64, param.numel())].clone()
        param.data.add_(update.to(param.data.dtype), alpha=-learning_rate)
        diagnostics["numel"] += param.numel()
        probe_after = param.data.reshape(-1)[:probe_before.numel()]
        observed_probe = probe_after.float() - probe_before.float()
        expected_probe = delta.float().reshape(-1)[:probe_before.numel()]
        roundoff_probe = observed_probe - expected_probe
        diagnostics["probe_numel"] += probe_before.numel()
        diagnostics["probe_expected_delta_sq"] += float(expected_probe.square().sum())
        diagnostics["probe_observed_delta_sq"] += float(observed_probe.square().sum())
        diagnostics["probe_roundoff_max_abs"] = max(
            diagnostics["probe_roundoff_max_abs"], float(roundoff_probe.abs().max())
        )
        diagnostics["probe_expected_nonzero_observed_zero"] += int(
            ((expected_probe != 0) & (observed_probe == 0)).sum()
        )
        diagnostics["g_sq"] += float(torch.dot(g.ravel(), g.ravel()))
        diagnostics["m_sq"] += float(torch.dot(m_for_update.ravel(), m_for_update.ravel()))
        diagnostics["m_raw_sq"] += float(torch.dot(m.ravel(), m.ravel()))
        diagnostics["m_prev_sq"] += float(torch.dot(m_previous.ravel(), m_previous.ravel()))
        diagnostics["m_prev_g_dot"] += float(torch.dot(m_previous.ravel(), g.ravel()))
        diagnostics["m_prev_m_dot"] += float(torch.dot(m_previous.ravel(), m.ravel()))
        nonzero = (m_previous != 0) & (g != 0)
        diagnostics["sign_agree"] += int((((m_previous > 0) & (g > 0)) | ((m_previous < 0) & (g < 0))).sum())
        diagnostics["sign_valid"] += int(nonzero.sum())
        diagnostics["sqrt_v_sq"] += float(torch.dot(sqrt_v.ravel(), sqrt_v.ravel()))
        diagnostics["update_sq"] += float(torch.dot(update.ravel(), update.ravel()))
        diagnostics["update_max_abs"] = max(
            diagnostics["update_max_abs"], float(update.abs().max())
        )
        delta_float = delta.float()
        diagnostics["actual_delta_sq"] += float(torch.dot(delta_float.ravel(), delta_float.ravel()))
        diagnostics["actual_delta_max_abs"] = max(
            diagnostics["actual_delta_max_abs"], float(delta_float.abs().max())
        )
    return state


@torch.no_grad()
def adaptive_fzoo_decoupled_update(
    named_parameters,
    zo_random_seeds,
    projected_grad,
    learning_rate,
    beta1,
    beta2,
    eps,
    state,
    direction_kind,
    norm_kind,
    direction="rademacher",
    bias_correction=False,
    step=None,
    diagnostics=None,
):
    """Use m/sqrt(v_m) or m/sqrt(v_g) as direction and match the other's L2.

    This is a diagnostic 2x2 factorization, not a memory-efficient optimizer:
    every parameter keeps m, v_m, and v_g. Norm matching is global over all
    trainable parameters, preserving relative tensor scales within a direction.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if direction not in ZO_DIRECTIONS:
        raise ValueError(f"Unknown ZO direction distribution: {direction}")
    if direction_kind not in ZO_SECOND_MOMENTS or norm_kind not in ZO_SECOND_MOMENTS:
        raise ValueError("Decoupled direction and norm must be m2 or g2")
    if len(zo_random_seeds) != len(projected_grad) or not zo_random_seeds:
        raise ValueError("FZOO seeds and projected gradients must be non-empty and aligned")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("FZOO moment coefficients must be in [0, 1)")
    if bias_correction and (step is None or step < 1):
        raise ValueError("Bias correction requires a 1-based optimizer step")
    if not named_parameters:
        raise ValueError("Adaptive FZOO requires at least one trainable parameter")
    if state is None:
        state = [
            (
                torch.zeros_like(param, dtype=torch.float32),
                torch.zeros_like(param, dtype=torch.float32),
                torch.zeros_like(param, dtype=torch.float32),
            )
            for _, param in named_parameters
        ]
    if len(state) != len(named_parameters):
        raise ValueError("Decoupled FZOO state does not match trainable parameters")

    if diagnostics is not None:
        diagnostics.update(
            numel=0,
            g_sq=0.0,
            m_sq=0.0,
            m_raw_sq=0.0,
            sqrt_v_sq=0.0,
            update_sq=0.0,
            update_max_abs=0.0,
            actual_delta_sq=0.0,
            actual_delta_max_abs=0.0,
            probe_numel=0,
            probe_expected_delta_sq=0.0,
            probe_observed_delta_sq=0.0,
            probe_roundoff_max_abs=0.0,
            probe_expected_nonzero_observed_zero=0,
            m_prev_sq=0.0,
            m_prev_g_dot=0.0,
            m_prev_m_dot=0.0,
            sign_agree=0,
            sign_valid=0,
            direction_m2_sq=0.0,
            direction_g2_sq=0.0,
            direction_dot=0.0,
        )

    generators = [
        torch.Generator(device=named_parameters[0][1].data.device).manual_seed(int(seed))
        for seed in zo_random_seeds
    ]
    direction_m2_sq = 0.0
    direction_g2_sq = 0.0
    for (name, param), (m, v_m, v_g) in zip(named_parameters, state):
        g = torch.zeros_like(param, dtype=torch.float32)
        for generator, alpha in zip(generators, projected_grad):
            z = sample_zo_direction(param, direction, generator)
            g.add_(z.float(), alpha=float(alpha))
        m_previous = m.clone() if diagnostics is not None else None
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        v_m.mul_(beta2).addcmul_(m, m, value=1 - beta2)
        v_g.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        if bias_correction:
            m_for_update = m / (1 - beta1 ** step)
            correction_v = 1 - beta2 ** step
            sqrt_v_m = (v_m / correction_v).sqrt()
            sqrt_v_g = (v_g / correction_v).sqrt()
        else:
            m_for_update = m
            sqrt_v_m = v_m.sqrt()
            sqrt_v_g = v_g.sqrt()
        update_m = m_for_update / (sqrt_v_m + eps)
        update_g = m_for_update / (sqrt_v_g + eps)
        direction_m2_sq += float(torch.dot(update_m.ravel(), update_m.ravel()))
        direction_g2_sq += float(torch.dot(update_g.ravel(), update_g.ravel()))

        if diagnostics is not None:
            diagnostics["g_sq"] += float(torch.dot(g.ravel(), g.ravel()))
            diagnostics["m_sq"] += float(torch.dot(m_for_update.ravel(), m_for_update.ravel()))
            diagnostics["m_raw_sq"] += float(torch.dot(m.ravel(), m.ravel()))
            diagnostics["m_prev_sq"] += float(torch.dot(m_previous.ravel(), m_previous.ravel()))
            diagnostics["m_prev_g_dot"] += float(torch.dot(m_previous.ravel(), g.ravel()))
            diagnostics["m_prev_m_dot"] += float(torch.dot(m_previous.ravel(), m.ravel()))
            nonzero = (m_previous != 0) & (g != 0)
            diagnostics["sign_agree"] += int((((m_previous > 0) & (g > 0)) | ((m_previous < 0) & (g < 0))).sum())
            diagnostics["sign_valid"] += int(nonzero.sum())
            diagnostics["direction_dot"] += float(torch.dot(update_m.ravel(), update_g.ravel()))

    direction_sq = direction_m2_sq if direction_kind == "m2" else direction_g2_sq
    target_sq = direction_m2_sq if norm_kind == "m2" else direction_g2_sq
    norm_scale = (target_sq / direction_sq) ** 0.5 if direction_sq > 0 else 0.0
    if diagnostics is not None:
        diagnostics["direction_m2_sq"] = direction_m2_sq
        diagnostics["direction_g2_sq"] = direction_g2_sq
        diagnostics["norm_scale"] = norm_scale

    for (name, param), (m, v_m, v_g) in zip(named_parameters, state):
        if bias_correction:
            m_for_update = m / (1 - beta1 ** step)
            correction_v = 1 - beta2 ** step
        else:
            m_for_update = m
            correction_v = 1.0
        if direction_kind == "m2":
            sqrt_v = (v_m / correction_v).sqrt()
        else:
            sqrt_v = (v_g / correction_v).sqrt()
        update = (m_for_update / (sqrt_v + eps)) * norm_scale
        if diagnostics is None:
            param.data.add_(update.to(param.data.dtype), alpha=-learning_rate)
            continue

        delta = update.to(param.data.dtype) * (-learning_rate)
        probe_before = param.data.reshape(-1)[:min(64, param.numel())].clone()
        param.data.add_(update.to(param.data.dtype), alpha=-learning_rate)
        diagnostics["numel"] += param.numel()
        probe_after = param.data.reshape(-1)[:probe_before.numel()]
        observed_probe = probe_after.float() - probe_before.float()
        expected_probe = delta.float().reshape(-1)[:probe_before.numel()]
        roundoff_probe = observed_probe - expected_probe
        diagnostics["probe_numel"] += probe_before.numel()
        diagnostics["probe_expected_delta_sq"] += float(expected_probe.square().sum())
        diagnostics["probe_observed_delta_sq"] += float(observed_probe.square().sum())
        diagnostics["probe_roundoff_max_abs"] = max(
            diagnostics["probe_roundoff_max_abs"], float(roundoff_probe.abs().max())
        )
        diagnostics["probe_expected_nonzero_observed_zero"] += int(
            ((expected_probe != 0) & (observed_probe == 0)).sum()
        )
        diagnostics["sqrt_v_sq"] += float(torch.dot(sqrt_v.ravel(), sqrt_v.ravel()))
        diagnostics["update_sq"] += float(torch.dot(update.ravel(), update.ravel()))
        diagnostics["update_max_abs"] = max(
            diagnostics["update_max_abs"], float(update.abs().max())
        )
        delta_float = delta.float()
        diagnostics["actual_delta_sq"] += float(torch.dot(delta_float.ravel(), delta_float.ravel()))
        diagnostics["actual_delta_max_abs"] = max(
            diagnostics["actual_delta_max_abs"], float(delta_float.abs().max())
        )
    return state
