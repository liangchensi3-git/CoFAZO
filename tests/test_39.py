"""Independent all-history oracle, bounded allocations and FFN observer checks."""
import argparse,json,sys,tempfile,unittest
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'src'))
import test_r34 as ref
from test_rc4 import capture_codec_inputs,verify_stored_history
from full_fzoo_fast import FullFZOO
from r39_all_m import AllMatrixRC4
from r39_observer import attach
from r39_fingerprint import fingerprint
DEVICE='cpu';PRODUCTION=False

def create(named,chunk=127):
    return AllMatrixRC4(named,'rc',m_compression='signed_rc4',q=8,chunk=chunk,no_history='none',drop_masked_m=False,diagnostics_interval=2)

class AllHistoryTests(unittest.TestCase):
    def setUp(self):ref.DEVICE=DEVICE

    def test_first_update_matches_full_history_not_selective_history(self):
        for dtype in (torch.float16,torch.float32):
            a=create(ref.parameters(dtype));b=FullFZOO(ref.parameters(dtype),'rc',q=8,chunk=127,no_history='none',diagnostics_interval=2)
            a.update(list(range(8)),[.03*j-.1 for j in range(8)],1e-5);b.update(list(range(8)),[.03*j-.1 for j in range(8)],1e-5)
            for (_,p),(_,q) in zip(a.named,b.named):self.assertTrue(torch.equal(p,q))
            for va,vb in zip(a.v,b.v):
                for p,q in zip(va if isinstance(va,tuple) else (va,),vb if isinstance(vb,tuple) else (vb,)):self.assertTrue(torch.equal(p,q))
            self.assertFalse(any(a.no_history_mask));self.assertTrue(all(m is not None for m in a.m))

    def oracle(self,named,chunk,steps=3):
        opt=create(named,chunk);capture=capture_codec_inputs(opt)
        pp=[(n,torch.nn.Parameter(p.detach().clone())) for n,p in named]
        direction=FullFZOO(pp,'rc',q=8,chunk=chunk,no_history='none')
        hist=[torch.zeros_like(p,dtype=torch.float32) for n,p in pp]
        vv=[tuple(t.clone() for t in v) if isinstance(v,tuple) else v.clone() for v in direction.v]
        for step in range(steps):
            seeds=list(range(step*8,step*8+8));coeff=[.13,-.2,.05,-.07,.11,.03,-.17,.06]
            gg=[torch.zeros_like(p,dtype=torch.float32) for n,p in pp]
            for k,start,p,zs in direction.blocks(seeds):
                for c,z in zip(coeff,zs):gg[k].view(-1)[start:start+p.numel()].add_(z,alpha=c)
            opt.update(seeds,coeff,1e-5)
            with torch.no_grad():
                for k,((name,p),g) in enumerate(zip(pp,gg)):
                    current=hist[k].mul(.9).add_(g,alpha=.1);v=vv[k]
                    if isinstance(v,tuple):
                        v[0].mul_(.99).add_(current.square().mean(1),alpha=.01)
                        v[1].mul_(.99).add_(current.square().mean(0),alpha=.01)
                        den=v[0][:,None]*v[1][None,:]/v[0].mean().clamp_min(1e-30)
                    else:v.mul_(.99).add_(current.square(),alpha=.01);den=v
                    p.add_(current/(den+1e-8).sqrt(),alpha=-1e-5)
                    torch.testing.assert_close(opt.named[k][1],p,rtol=1e-4,atol=2e-6)
                    for a,b in zip(opt.v[k] if isinstance(v,tuple) else (opt.v[k],),v if isinstance(v,tuple) else (v,)):torch.testing.assert_close(a,b,rtol=1e-4,atol=3e-7)
                    if p.ndim==2:
                        torch.testing.assert_close(capture[k],current,rtol=1e-6,atol=1e-7)
                        hist[k],_=verify_stored_history(opt.m[k],capture[k])
                    else:hist[k]=current;torch.testing.assert_close(opt.m[k],current)

    def test_multistep_all_matrix_dense_oracle(self):
        for dtype in (torch.float16,torch.float32):
            for chunk in (127,1024):self.oracle(ref.parameters(dtype,rows=37,cols=25),chunk)

    def test_production_embedding_attention_chunk_tail(self):
        rows,cols,chunk=(2305,2048,4194304) if DEVICE.startswith('cuda') or PRODUCTION else (137,32,2048)
        named=[(name,torch.nn.Parameter(torch.linspace(-.2,.3,rows*cols,device=DEVICE,dtype=torch.float32).reshape(rows,cols))) for name in ['model.embed_tokens.weight','model.layers.0.self_attn.q_proj.weight']]
        self.oracle(named,chunk,2)
        print('R39_ALL_HISTORY_PRODUCTION '+json.dumps(dict(shape=[rows,cols],steps=2,modules=['embedding','attention'])),flush=True)

    def test_global_rng_and_forward_query_count(self):
        opt=create(ref.parameters(torch.float16));before=torch.cuda.get_rng_state() if DEVICE.startswith('cuda') else torch.get_rng_state();calls=[]
        def loss():
            value=sum((p.float()*torch.arange(1,p.numel()+1,device=DEVICE).reshape(p.shape)).sum() for n,p in opt.named);calls.append(float(value));return value
        opt.step(loss,list(range(8)),1e-5);self.assertEqual(len(calls),9)
        after=torch.cuda.get_rng_state() if DEVICE.startswith('cuda') else torch.get_rng_state();self.assertTrue(torch.equal(before,after))

    def test_no_dense_embedding_or_attention_m_allocation(self):
        named=[(n,torch.nn.Parameter(torch.zeros(257,128,device=DEVICE))) for n in ['model.embed_tokens.weight','model.layers.0.self_attn.q_proj.weight','model.layers.0.mlp.gate_proj.weight']]
        weights={p.untyped_storage().data_ptr() for n,p in named}
        class Guard(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                out=func(*args,**(kwargs or {}))
                def check(x):
                    if torch.is_tensor(x) and x.is_floating_point() and x.numel()>4096 and x.untyped_storage().data_ptr() not in weights:raise AssertionError('Dense restored m allocation '+str(func))
                    if isinstance(x,(tuple,list)):
                        for v in x:check(v)
                check(out);return out
        with Guard():
            opt=create(named,4096);opt.update(list(range(8)),[.1]*8,1e-5);opt.update(list(range(8,16)),[-.1]*8,1e-5)
        self.assertEqual(opt.m_bytes(),3*(257*64+4*(257+128)))

    def test_observer_ffn_only_and_no_numerical_effect(self):
        a=create(ref.parameters());b=create(ref.parameters())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'obs.jsonl';attach(b,path,17,2)
            for i in range(3):
                a.update(list(range(i*8,i*8+8)),[.1]*8,1e-5);b.update(list(range(i*8,i*8+8)),[.1]*8,1e-5)
            fa=fingerprint(a,Path(temp)/'a.json');fb=fingerprint(b,Path(temp)/'b.json');self.assertEqual(fa['items'],fb['items'])
            rows=[json.loads(x) for x in path.read_text('utf8').splitlines()]
            self.assertEqual(len(rows[0]['modules']),3);self.assertTrue(all('.mlp.' in x['name'] for x in rows[0]['modules']))

    def test_stateless_and_full_m_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            for kind in ('fzoo','rc'):
                opt=FullFZOO(ref.parameters(),kind,q=8,chunk=127,no_history='none')
                opt.update(list(range(8)),[.1]*8,1e-5)
                fp=fingerprint(opt,Path(temp)/(kind+'.json'))
                self.assertEqual(fp['total_bytes'],sum(p.numel()*p.element_size() for n,p in opt.named)+opt.state_bytes())
                self.assertEqual(any(x['key'].startswith('m/') for x in fp['items']),kind=='rc')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-cuda',action='store_true');p.add_argument('--production-shape',action='store_true');a,rest=p.parse_known_args()
    if a.require_cuda:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        DEVICE='cuda:0'
    PRODUCTION=a.production_shape;torch.set_num_threads(4);unittest.main(argv=[sys.argv[0],*rest])
