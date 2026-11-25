"""Dense mathematical oracles and allocation/RNG checks for approximate history."""
import argparse
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'src'))
from compressed_fzoo import CompressedFZOO
from full_fzoo_r33_reference import FullFZOO as R33
from m_compression import pack_signs,unpack_signs,SignedRC,LowRank
DEVICE='cpu'


def parameters(dtype=torch.float32,rows=37,cols=24):
    shapes=[('model.embed_tokens.weight',(17,cols)),
        ('model.layers.0.mlp.gate_proj.weight',(rows,cols)),
        ('model.layers.0.mlp.up_proj.weight',(rows,cols)),
        ('model.layers.0.mlp.down_proj.weight',(cols,rows)),
        ('model.layers.0.self_attn.q_proj.weight',(cols,cols)),
        ('model.layers.0.self_attn.k_proj.weight',(8,cols)),
        ('model.layers.0.self_attn.v_proj.weight',(8,cols)),
        ('model.layers.0.self_attn.o_proj.weight',(cols,cols)),
        ('model.layers.0.self_attn.q_norm.weight',(8,)),('model.norm.weight',(cols,))]
    return [(n,torch.nn.Parameter(torch.linspace(-.2,.3,torch.Size(s).numel(),device=DEVICE,dtype=dtype).reshape(s))) for n,s in shapes]


def optimizer(named,mode,chunk=127,rank=16):
    return CompressedFZOO(named,'rc',m_compression=mode,rank=rank,q=8,chunk=chunk,mu=5e-4,
        no_history='attention_embedding',drop_masked_m=True,diagnostics_interval=2)


class DenseOracle:
    """Full tensors allowed only in this test, independently compressing each A."""
    def __init__(self,named,mode,chunk,rank=16):
        self.named=named;self.mode=mode;self.rank=rank;self.step=0
        self.base=R33(named,'rc',q=8,chunk=chunk,no_history='attention_embedding',drop_masked_m=True)
        self.history=[torch.zeros_like(p,dtype=torch.float32) for n,p in named]
        self.v=[tuple(t.clone() for t in v) if isinstance(v,tuple) else v.clone() for v in self.base.v]
        self.last_a=[]

    @torch.no_grad()
    def update(self,seeds,coeff,lr):
        self.step+=1
        gs=[torch.zeros_like(p,dtype=torch.float32) for n,p in self.named]
        for k,start,p,zs in self.base.blocks(seeds):
            dst=gs[k].view(-1)[start:start+p.numel()]
            for c,z in zip(coeff,zs):dst.add_(z,alpha=c)
        errors=norms=0.
        self.last_a=[]
        for k,((name,p),g) in enumerate(zip(self.named,gs)):
            current=g if self.base.no_history_mask[k] else self.history[k]*.9+g*.1
            self.last_a.append(current)
            v=self.v[k]
            if isinstance(v,tuple):
                v[0].mul_(.99).add_(current.square().mean(1),alpha=.01)
                v[1].mul_(.99).add_(current.square().mean(0),alpha=.01)
                denominator=v[0][:,None]*v[1][None,:]/v[0].mean().clamp_min(1e-30)
            else:
                v.mul_(.99).add_(current.square(),alpha=.01);denominator=v
            p.add_(current/torch.sqrt(denominator+1e-8),alpha=-lr)
            if '.mlp.' in name:
                if self.mode=='signed_rc':
                    r=current.abs().mean(1);c=current.abs().mean(0)
                    signs=torch.where(current>=0,1.,-1.)
                    self.history[k]=signs*r[:,None]*c[None,:]/r.mean().clamp_min(1e-30)
                else:
                    rank=min(self.rank,*p.shape)
                    seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')%(2**63-1)
                    gen=torch.Generator(device=p.device).manual_seed((seed+104729*self.step)%(2**63-1))
                    omega=torch.randn((p.shape[1],rank),generator=gen,device=p.device,dtype=torch.float32)
                    u,_=torch.linalg.qr(current@omega,mode='reduced')
                    self.history[k]=u@(u.T@current)
                norms+=float(current.double().square().sum())
                errors+=float((current.double()-self.history[k].double()).square().sum())
            else:self.history[k]=current.clone()
        return (errors/max(norms,1e-30))**.5


class R34Tests(unittest.TestCase):
    def test_pack_sign_roundtrip_nonbyte_width_and_zero(self):
        for width in (1,7,8,13,32):
            a=torch.arange(3*width,device=DEVICE).reshape(3,width).float().sub_(5)
            bits=pack_signs(a)
            self.assertEqual(bits.dtype,torch.uint8);self.assertEqual(tuple(bits.shape),(3,(width+7)//8))
            self.assertTrue(torch.equal(unpack_signs(bits,width),torch.where(a>=0,1.,-1.)))

    def test_multistep_dense_oracle_m_v_parameters_and_error(self):
        for mode in ('signed_rc','lowrank'):
            for dtype in (torch.float16,torch.float32):
                for chunk in (127,1024):
                    pp=parameters(dtype);qq=parameters(dtype)
                    opt=optimizer(pp,mode,chunk);ref=DenseOracle(qq,mode,chunk)
                    capacity=opt.state_bytes()
                    for step in range(4):
                        seeds=[step*113+j+3 for j in range(8)];coeff=[.13,-.2,.05,-.07,.11,.03,-.17,.06]
                        opt.update(seeds,coeff,1e-5);error=ref.update(seeds,coeff,1e-5)
                        for k,((_,a),(_,b)) in enumerate(zip(pp,qq)):
                            torch.testing.assert_close(a,b,rtol=2e-3 if dtype==torch.float16 else 1e-5,atol=1e-4 if dtype==torch.float16 else 2e-7)
                            for x,y in zip(opt.v[k] if isinstance(opt.v[k],tuple) else (opt.v[k],),ref.v[k] if isinstance(ref.v[k],tuple) else (ref.v[k],)):
                                torch.testing.assert_close(x,y,rtol=3e-5,atol=2e-8)
                            state=opt.m[k]
                            if getattr(state,'compressed',False):torch.testing.assert_close(state.decode(0,a.shape[0]),ref.history[k],rtol=2e-4,atol=2e-6)
                            elif state is not None:torch.testing.assert_close(state,ref.history[k])
                        if opt.last_diagnostics:self.assertAlmostEqual(opt.compression_diagnostics['relative_frobenius_error'],error,places=4)
                        else:self.assertIsNone(opt.compression_diagnostics)
                        self.assertEqual(opt.state_bytes(),capacity)

    def test_first_update_matches_r33_current_m_not_new_compressed_m(self):
        for mode in ('signed_rc','lowrank'):
            pp=parameters();qq=parameters();opt=optimizer(pp,mode)
            ref=R33(qq,'rc',q=8,chunk=127,no_history='attention_embedding',drop_masked_m=True)
            opt.update(list(range(8)),[.03*j-.1 for j in range(8)],1e-5)
            ref.update(list(range(8)),[.03*j-.1 for j in range(8)],1e-5)
            for (_,a),(_,b) in zip(pp,qq):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
            for x,y in zip(opt.v,ref.v):
                for a,b in zip(x if isinstance(x,tuple) else (x,),y if isinstance(y,tuple) else (y,)):torch.testing.assert_close(a,b)

    def test_zero_history_and_zero_gradient_are_finite(self):
        for mode in ('signed_rc','lowrank'):
            opt=optimizer(parameters(),mode)
            opt.update(list(range(8)),[0.]*8,1e-5)
            for m in opt.m:
                if getattr(m,'compressed',False):self.assertEqual(float(m.decode(0,m.rows).abs().max()),0.)
            self.assertEqual(opt.compression_diagnostics['relative_frobenius_error'],0.)

    def test_global_rng_untouched_and_directions_identical(self):
        for mode in ('signed_rc','lowrank'):
            before=torch.cuda.get_rng_state() if DEVICE.startswith('cuda') else torch.get_rng_state()
            opt=optimizer(parameters(),mode)
            opt.update(list(range(8)),[.1]*8,1e-5)
            after=torch.cuda.get_rng_state() if DEVICE.startswith('cuda') else torch.get_rng_state()
            self.assertTrue(torch.equal(before,after))
            ref=R33(parameters(),'rc',q=8,chunk=127,no_history='attention_embedding',drop_masked_m=True)
            for a,b in zip(opt.blocks(list(range(8))),ref.blocks(list(range(8)))):
                for x,y in zip(a[3],b[3]):self.assertTrue(torch.equal(x,y))

    def test_no_dense_ffn_allocation_constructor_or_updates(self):
        class AllocationAudit(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                out=func(*args,**(kwargs or {}))
                def visit(value):
                    if torch.is_tensor(value) and value.dtype in (torch.float32,torch.float64) and value.numel()>=257*128 and value.untyped_storage().data_ptr() not in weight_storages:
                        raise AssertionError('Full FFN-sized floating allocation: '+str(func))
                    if isinstance(value,(list,tuple)):
                        for x in value:visit(x)
                visit(out);return out
        for mode in ('signed_rc','lowrank'):
            pp=parameters(rows=257,cols=128)
            weight_storages={p.untyped_storage().data_ptr() for n,p in pp}
            with AllocationAudit():
                opt=optimizer(pp,mode,chunk=4096)
                opt.update(list(range(8)),[.1]*8,1e-5)
                opt.update(list(range(8,16)),[-.1]*8,1e-5)
            self.assertFalse(any(key in vars(opt) for key in ('pending','states','history','gradient_cache')))

    def test_state_accounting_bits_and_factors_no_dense_copies(self):
        for mode in ('signed_rc','lowrank'):
            pp=parameters();opt=optimizer(pp,mode)
            meta=opt.metadata();total=0
            for (name,p),m in zip(pp,opt.m):
                if opt.no_history_mask[pp.index((name,p))]:self.assertIsNone(m)
                elif '.mlp.' in name:
                    a,b=p.shape
                    total+=a*((b+7)//8)+4*(a+b) if mode=='signed_rc' else 4*min(16,a,b)*(a+b)
                    self.assertTrue(all(t.device==p.device for t in m.tensors()))
                else:total+=p.numel()*4
            self.assertEqual(opt.m_bytes(),total)
            self.assertEqual(meta['compressed_m_bytes']+meta['dense_m_bytes'],total)

    def test_full_forward_count_and_exception_restoration(self):
        for mode in ('signed_rc','lowrank'):
            pp=parameters(torch.float32);opt=optimizer(pp,mode);calls=[0]
            def closure():
                calls[0]+=1
                return sum((p*torch.arange(1,p.numel()+1,device=DEVICE).reshape(p.shape)).sum() for n,p in pp)
            opt.step(closure,list(range(8)),1e-5);self.assertEqual(calls[0],9)
            before=[p.clone() for n,p in pp];counter=[0]
            def fail():
                counter[0]+=1
                if counter[0]==2:raise RuntimeError('injected objective failure')
                return closure()
            with self.assertRaises(RuntimeError):opt.step(fail,list(range(8)),1e-5)
            for (_,p),old in zip(pp,before):torch.testing.assert_close(p,old,rtol=1e-6,atol=3e-8)

    def test_lowrank_full_rank_recovers_history(self):
        pp=parameters(rows=13,cols=8);opt=optimizer(pp,'lowrank',rank=16)
        ref=DenseOracle(parameters(rows=13,cols=8),'lowrank',127,16)
        for step in range(2):
            seeds=list(range(step*8,step*8+8));coeff=[.1*j-.23 for j in range(8)]
            opt.update(seeds,coeff,1e-5);ref.update(seeds,coeff,1e-5)
            for k,m in enumerate(opt.m):
                if getattr(m,'compressed',False):torch.testing.assert_close(m.decode(0,m.rows),ref.last_a[k],rtol=1e-4,atol=1e-6)

    def test_production_chunk_rng_boundary(self):
        rows,cols,chunk=(2305,2048,4194304) if DEVICE.startswith('cuda') else (137,32,2048)
        for mode in ('signed_rc','lowrank'):
            # One large FFN is enough to cover full-size plus partial CUDA draws.
            pp=parameters(rows=rows,cols=cols)[:2]+parameters(rows=3,cols=cols)[4:5]
            qq=[(n,torch.nn.Parameter(p.clone())) for n,p in pp]
            opt=optimizer(pp,mode,chunk);ref=DenseOracle(qq,mode,chunk)
            for step in range(2):
                seeds=list(range(step*8,step*8+8));coeff=[.1*j-.23 for j in range(8)]
                opt.update(seeds,coeff,1e-5);ref.update(seeds,coeff,1e-5)
                for (_,p),(_,old) in zip(pp,qq):torch.testing.assert_close(p,old,rtol=1e-4,atol=2e-6)
                for a,b in zip(opt.v,ref.v):
                    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-4,atol=3e-7)


    def test_invalid_modes_are_rejected(self):
        with self.assertRaises(ValueError):optimizer(parameters(),'unknown')
        with self.assertRaises(ValueError):optimizer(parameters(),'lowrank',rank=0)
        with self.assertRaises(ValueError):CompressedFZOO(parameters(),'rc',m_compression='signed_rc')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-cuda',action='store_true');args,rest=parser.parse_known_args()
    if args.require_cuda:
        if not torch.cuda.is_available():raise RuntimeError('CUDA cannot skip')
        DEVICE='cuda:0'
    torch.set_num_threads(4)
    unittest.main(argv=[sys.argv[0],*rest])
