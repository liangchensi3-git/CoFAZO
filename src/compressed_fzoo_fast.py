"""R34: streaming FFN history compression on the frozen R33 selective mask."""
import torch
from full_fzoo_fast import FullFZOO
from momentum_mask import resolve_mask
from m_compression import SignedRC,LowRank,tensor_bytes
from rc4_m import SignedRC4


class CompressedFZOO(FullFZOO):
    def __init__(self,named,kind,*,m_compression,rank=16,**kwargs):
        if m_compression not in ('signed_rc','signed_rc4','lowrank') or rank<1:
            raise ValueError('Invalid FFN compression/rank')
        if kind!='rc' or kwargs.get('no_history')!='attention_embedding' or not kwargs.get('drop_masked_m'):
            raise ValueError('R34 requires the R33 selective-m rc base')
        named=list(named)
        ffn=resolve_mask(named,'ffn')
        if not any(ffn): raise ValueError('No FFN matrices found')
        state={'signed_rc':SignedRC,'signed_rc4':SignedRC4,'lowrank':LowRank}[m_compression]
        def factory(k,name,p):
            if ffn[k]:
                if p.ndim!=2: raise ValueError('Only FFN matrices can be compressed')
                return state(name,p,rank)
            return torch.zeros_like(p,dtype=torch.float32)
        super().__init__(named,kind,m_factory=factory,**kwargs)
        self.compression=m_compression;self.rank=rank;self.ffn_mask=ffn
        self.compression_diagnostics=None

    def m_tensors(self):
        return [t for m in self.m if m is not None for t in (m.tensors() if getattr(m,'compressed',False) else (m,))]

    def m_bytes(self): return tensor_bytes(self.m_tensors())

    def state_bytes(self):
        return self.m_bytes()+tensor_bytes([t for v in self.v for t in (v if isinstance(v,tuple) else (v,))])

    def metadata(self):
        return dict(m_compression=self.compression,compression_rank=self.rank if self.compression=='lowrank' else None,
            compressed_names=[n for (n,p),take in zip(self.named,self.ffn_mask) if take],
            compressed_numel=sum(p.numel() for (n,p),take in zip(self.named,self.ffn_mask) if take),
            compressed_m_bytes=tensor_bytes([t for m in self.m if getattr(m,'compressed',False) for t in m.tensors()]),
            dense_m_bytes=tensor_bytes([m for m in self.m if torch.is_tensor(m)]),
            m_state_tensors=[dict(name=n,shape=list(p.shape),mode=m.mode if getattr(m,'compressed',False) else ('dense' if m is not None else 'absent'),
                bytes=tensor_bytes(m.tensors() if getattr(m,'compressed',False) else (() if m is None else (m,))))
                for (n,p),m in zip(self.named,self.m)],
            history_compression_only=True,cpu_optimizer_offload=False,
            sketch_policy='independent_name_step_seed_thin_qr' if self.compression=='lowrank' else None)

    @torch.no_grad()
    def update(self,seeds,coeff,lr):
        diagnostic=self.updates%self.diagnostics_interval==0
        self.last_diagnostics=diagnostic;self.updates+=1
        scalar=lambda:torch.zeros((),device=self.device,dtype=torch.float64)
        g2,delta2,theta2=(scalar(),scalar(),scalar()) if diagnostic else (None,None,None)
        moment2,error2=(scalar(),scalar()) if diagnostic else (None,None)
        finite=torch.ones((),device=self.device,dtype=torch.bool)
        states={};pending={}
        def apply(p,value):
            before=p.clone() if diagnostic else None
            p.add_(value,alpha=-lr)
            finite.logical_and_(torch.isfinite(p).all())
            if diagnostic:
                delta2.add_((p.double()-before.double()).square().sum())
                theta2.add_(before.double().square().sum())
        for v in self.v:
            if isinstance(v,tuple): v[0].mul_(self.b2);v[1].mul_(self.b2)
        # Pass 1: form present A, accumulate SAME A^2 into v, prepare compression.
        for k,start,p,zs in self.blocks(seeds,capture_states=states):
            g=torch.zeros_like(p,dtype=torch.float32)
            for c,z in zip(coeff,zs):g.add_(z,alpha=c)
            if diagnostic:g2.add_(g.double().square().sum())
            state=self.m[k];shape=self.named[k][1].shape
            compressed=getattr(state,'compressed',False)
            if compressed:
                cols=shape[1];row=start//cols
                if start==0:pending[k]=state.begin(self.updates)
                moment=state.combine(g.view(-1,cols),row,self.b1)
                state.accumulate(pending[k],moment,row)
                if diagnostic:moment2.add_(moment.double().square().sum())
                if start+p.numel()==self.named[k][1].numel():state.finish_first(pending[k])
                moment=moment.view(-1)
            elif state is None:moment=g
            else:
                moment=state.view(-1)[start:start+p.numel()]
                moment.mul_(self.b1).add_(g,alpha=1-self.b1)
            square=moment.square();v=self.v[k]
            if isinstance(v,tuple):
                square=square.view(-1,shape[1]);row=start//shape[1]
                v[0][row:row+square.shape[0]].add_(square.mean(1),alpha=1-self.b2)
                v[1].add_(square.sum(0),alpha=(1-self.b2)/shape[0])
            else:v.view(-1)[start:start+p.numel()].mul_(self.b2).add_(square,alpha=1-self.b2)
        # Release the last first-pass direction list before reconstruction.
        del g,zs,moment,square
        # Pass 2: reconstruct CURRENT g once; keep old history until consumed.
        streams=[torch.Generator(device=self.device) for _ in seeds]
        for k,start,p,zs in self.blocks([]):
            state=self.m[k];compressed=getattr(state,'compressed',False)
            shape=self.named[k][1].shape
            if state is None or compressed:
                if start==0:
                    for stream,saved in zip(streams,states[k]):stream.set_state(saved)
                g=torch.zeros_like(p,dtype=torch.float32)
                for c,stream in zip(coeff,streams):
                    z=torch.randint(0,2,(p.numel(),),generator=stream,device=p.device,dtype=torch.int32).float()
                    torch.add(-1.0,z,alpha=2,out=z)
                    g.add_(z,alpha=c)
                del z
                if compressed:
                    cols=shape[1];row=start//cols
                    matrix=state.combine(g.view(-1,cols),row,self.b1)
                    error=state.second(pending[k],matrix,row,diagnostic)
                    if diagnostic and error is not None:error2.add_(error)
                    moment=matrix.view(-1)
                else:moment=g
            else:moment=state.view(-1)[start:start+p.numel()]
            v=self.v[k]
            if isinstance(v,tuple):
                cols=shape[1];row=start//cols;rows=p.numel()//cols
                denom=(v[0][row:row+rows,None]*v[1][None,:]/v[0].mean().clamp_min(1e-30)).view(-1)
            else:denom=v.view(-1)[start:start+p.numel()]
            apply(p,moment/torch.sqrt(denom+self.zeta))
            if compressed and start+p.numel()==self.named[k][1].numel():
                if diagnostic and self.compression=='lowrank':
                    # Orthogonal projection residual: ||A-QC||^2=||A||^2-||C||^2.
                    error2.sub_(pending[k]['c'].double().square().sum())
                state.commit(pending.pop(k))
                finite.logical_and_(torch.stack([torch.isfinite(t).all() for t in state.tensors() if t.is_floating_point() or t.is_complex()]).all())
        if not bool(finite):raise FloatingPointError('Nonfinite compressed state or parameter update')
        self.compression_diagnostics=None
        if diagnostic:
            if self.compression=='lowrank':error2.add_(moment2)
            self.compression_diagnostics=dict(relative_frobenius_error=float((error2.clamp_min(0)/moment2.clamp_min(1e-30)).sqrt()),
                current_m_norm=float(moment2.sqrt()),scope='FFN history compression only',
                error_measure='projection_energy_identity' if self.compression=='lowrank' else 'direct_block_error')
            return float(g2.sqrt()),float(delta2.sqrt()),float(theta2.sqrt())
        return None,None,None

    @torch.no_grad()
    def step(self,closure,seeds,lr):
        result=super().step(closure,seeds,lr)
        result['compression_diagnostics']=self.compression_diagnostics
        return result
