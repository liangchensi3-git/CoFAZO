"""Reuse R37 four-bit updates; extend compressed history to every matrix."""
import torch
from full_fzoo_fast import FullFZOO
from compressed_fzoo_fast import CompressedFZOO
from rc4_m import SignedRC4

class AllMatrixRC4(CompressedFZOO):
    def __init__(self,named,kind,*,m_compression,rank=16,**kwargs):
        if kind!='rc' or m_compression!='signed_rc4' or kwargs.get('no_history')!='none' or kwargs.get('drop_masked_m'):
            raise ValueError('All-matrix RC4 requires complete history, rc v, and four-bit encoding')
        named=list(named)
        if any(p.ndim not in (1,2) for n,p in named):raise ValueError('Expected matrix/vector parameters')
        mask=[p.ndim==2 for n,p in named]
        if not any(mask):raise ValueError('No matrices to compress')
        def factory(k,name,p):
            return SignedRC4(name,p,rank) if mask[k] else torch.zeros_like(p,dtype=torch.float32)
        FullFZOO.__init__(self,named,kind,m_factory=factory,**kwargs)
        # Parent uses this mask for inventory only, not update arithmetic.
        self.ffn_mask=mask;self.compression='signed_rc4';self.rank=rank
        self.compression_diagnostics=None

    def metadata(self):
        return dict(super().metadata(),compression_scope='all_matrices')

    @torch.no_grad()
    def update(self,seeds,coeff,lr):
        result=super().update(seeds,coeff,lr)
        if self.compression_diagnostics is not None:
            self.compression_diagnostics['scope']='All matrix history compression only'
        return result
