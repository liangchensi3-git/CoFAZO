"""Streaming FZOO q+1 estimator and full/factored R-AdaZO states.

No whole-model direction/gradient/parameter clones. Perturbations use standard
in-place add/subtract (FP16/FP32 rounding is not exactly reversible). Fixed chunk
layout and isolated generators define the reproducible direction sequence.
"""
import math
import torch


class FullFZOO:
    def __init__(self, named, kind, q=8, mu=1e-3, chunk=1048576, beta1=.9, beta2=.99, zeta=1e-8, diagnostics_interval=100, no_history='none', drop_masked_m=False, m_factory=None):
        self.named=list(named); self.kind=kind; self.q=q; self.mu=mu; self.chunk=chunk
        self.b1=beta1; self.b2=beta2; self.zeta=zeta
        if kind not in ('fzoo','full','rc','momentum') or q<2: raise ValueError('Invalid method/q')
        if not self.named or any(p.dtype not in (torch.float16,torch.float32) or not p.is_contiguous() for _,p in self.named):
            raise ValueError('Contiguous FP16/FP32 parameters required')
        if diagnostics_interval<1:raise ValueError('diagnostics_interval must be positive')
        self.diagnostics_interval=diagnostics_interval;self.updates=0;self.last_diagnostics=False
        self.device=self.named[0][1].device
        self.d=sum(p.numel() for _,p in self.named)
        from momentum_mask import resolve_mask
        if no_history!='none' and kind!='rc': raise ValueError('History ablation requires rc')
        self.no_history=no_history
        self.no_history_mask=resolve_mask(self.named,no_history)
        self.drop_masked_m=bool(drop_masked_m)
        if self.drop_masked_m and no_history!='attention_embedding':
            raise ValueError('R33 state removal requires the attention+embedding mask')
        self.m=[]; self.v=[]
        for k,(_,p) in enumerate(self.named):
            keep_m=kind!='fzoo' and not (self.drop_masked_m and self.no_history_mask[k])
            self.m.append((m_factory(k,self.named[k][0],p) if m_factory else torch.zeros_like(p,dtype=torch.float32)) if keep_m else None)
            if kind in ('fzoo','momentum'): self.v.append(None)
            elif kind=='rc' and p.ndim==2:
                self.v.append((torch.zeros(p.shape[0],device=p.device,dtype=torch.float32),torch.zeros(p.shape[1],device=p.device,dtype=torch.float32)))
            else: self.v.append(torch.zeros_like(p,dtype=torch.float32))

    def blocks(self,seeds,capture_states=None):
        generators=[torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]
        for k,(_,p) in enumerate(self.named):
            if capture_states is not None and (self.no_history_mask[k] or getattr(self.m[k],'compressed',False)):
                # Small current-step PRNG metadata, not m/gradient offloading.
                # Capture at each selected tensor boundary; preserve the exact
                # original draw shapes and continue within that tensor later.
                capture_states[k]=tuple(g.get_state() for g in generators)
            width=p.shape[1] if p.ndim==2 else 1
            block=max(width,(self.chunk//width)*width)
            for start in range(0,p.numel(),block):
                size=min(block,p.numel()-start)
                zs=[]
                for generator in generators:
                    z=torch.randint(0,2,(size,),generator=generator,device=p.device,dtype=torch.int32).float()
                    torch.add(-1.0,z,alpha=2,out=z)
                    zs.append(z)
                yield k,start,p.view(-1)[start:start+size],zs

    @torch.no_grad()
    def perturb(self,seed,scale):
        for _,_,p,z in self.blocks([seed]): p.add_(z[0],alpha=self.mu*scale)

    @torch.no_grad()
    def update(self,seeds,coeff,lr):
        # Same phase as historical global_step % 100: updates 1,101,201,...
        diagnostic=self.updates%self.diagnostics_interval==0
        self.last_diagnostics=diagnostic
        self.updates+=1
        g2=torch.zeros((),device=self.device,dtype=torch.float64) if diagnostic else None
        delta2=torch.zeros_like(g2) if diagnostic else None
        theta2=torch.zeros_like(g2) if diagnostic else None
        finite=torch.ones((),device=self.device,dtype=torch.bool)
        replay_states={} if self.drop_masked_m else None
        def apply(p,update):
            before=p.clone() if diagnostic else None
            p.add_(update,alpha=-lr)
            finite.logical_and_(torch.isfinite(p).all())
            if diagnostic:
                delta2.add_((p.double()-before.double()).square().sum())
                theta2.add_(before.double().square().sum())
        # First pass accumulates m and row/column marginals of the SAME m^2.
        for k,(_,p) in enumerate(self.named):
            v=self.v[k]
            if isinstance(v,tuple):
                v[0].mul_(self.b2);v[1].mul_(self.b2)
        for k,start,p,zs in self.blocks(seeds,capture_states=replay_states):
            g=torch.zeros_like(p,dtype=torch.float32)
            for c,z in zip(coeff,zs):g.add_(z,alpha=c)
            if diagnostic:g2.add_(g.double().square().sum())
            if self.kind=='fzoo':
                apply(p,g)
                continue
            if self.m[k] is None:
                m=g  # Only the current block, never a full selected m slot.
            else:
                m=self.m[k].view(-1)[start:start+p.numel()]
                if self.no_history_mask[k]: m.copy_(g)
                else: m.mul_(self.b1).add_(g,alpha=1-self.b1)
            if self.kind=='momentum':
                apply(p,m)
                continue
            square=m.square();v=self.v[k];shape=self.named[k][1].shape
            if isinstance(v,tuple):
                square=square.view(-1,shape[1]);row=start//shape[1]
                v[0][row:row+square.shape[0]].add_(square.mean(1),alpha=1-self.b2)
                v[1].add_(square.sum(0),alpha=(1-self.b2)/shape[0])
            else: v.view(-1)[start:start+p.numel()].mul_(self.b2).add_(square,alpha=1-self.b2)
        # Adaptive arms use moments here, without regenerating directions.
        # R33 regenerates only CURRENT gradients for the dropped slots, after
        # all column statistics are complete. There is no history replay.
        if self.kind in ('full','rc'):
            streams=None
            for k,start,p,zs in self.blocks([]):
                if self.m[k] is None:
                    if start==0:
                        streams=[torch.Generator(device=self.device).set_state(state) for state in replay_states[k]]
                    current=torch.zeros_like(p,dtype=torch.float32)
                    for c,stream in zip(coeff,streams):
                        z=torch.randint(0,2,(p.numel(),),generator=stream,device=p.device,dtype=torch.int32).float().mul_(2).sub_(1)
                        current.add_(z,alpha=c)
                    moment=current
                else:
                    moment=self.m[k].view(-1)[start:start+p.numel()]
                v=self.v[k]
                if isinstance(v,tuple):
                    cols=self.named[k][1].shape[1]
                    row=start//cols;rows=p.numel()//cols
                    denom=(v[0][row:row+rows,None]*v[1][None,:]/v[0].mean().clamp_min(1e-30)).view(-1)
                else:denom=v.view(-1)[start:start+p.numel()]
                update=moment/torch.sqrt(denom+self.zeta)
                apply(p,update)
        if not bool(finite):raise FloatingPointError('Nonfinite parameter update')
        if diagnostic:return float(g2.sqrt()),float(delta2.sqrt()),float(theta2.sqrt())
        return None,None,None

    @torch.no_grad()
    def step(self,closure,seeds,lr):
        if len(seeds)!=self.q:raise ValueError('q seeds required')
        base=float(closure()); losses=[]
        if not math.isfinite(base):raise FloatingPointError('Invalid base loss')
        for seed in seeds:
            self.perturb(seed,1)
            try:
                loss=float(closure())
                if not math.isfinite(loss):raise FloatingPointError('Invalid perturbed loss')
                losses.append(loss)
            finally:self.perturb(seed,-1)
        sigma=float(torch.tensor(losses,dtype=torch.float64).std(unbiased=False))
        if sigma<1e-12:raise FloatingPointError('Degenerate perturbation loss standard deviation')
        coefficients=[(loss-base)/(self.q*sigma) for loss in losses]
        gn,dn,tn=self.update(seeds,coefficients,lr)
        if self.last_diagnostics and dn/max(tn,1e-30)>.1:raise FloatingPointError('Update exceeds 10% of model norm')
        return dict(loss=base,internal_last_loss=losses[-1],loss_difference=max(abs(x-base) for x in losses),
                    sigma=sigma,coefficients=coefficients,gradient_norm=gn,actual_delta_norm=dn,
                    parameter_norm=tn,relative_update_norm=dn/max(tn,1e-30) if self.last_diagnostics else None,
                    diagnostics_computed=self.last_diagnostics,state_bytes=self.state_bytes())

    def state_bytes(self):
        return sum(t.numel()*t.element_size() for x in self.m+self.v if x is not None for t in (x if isinstance(x,tuple) else (x,)))
