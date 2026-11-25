"""Independent dense quantization oracle and storage/observer/forward checks."""
import argparse,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'src'))
import test_r34 as reference
import test_observer
from rc4_m import SignedRC4,pack_codes,unpack_codes,encode,reconstruct
DEVICE='cpu'
FORCE_PRODUCTION_SHAPE=False

def rounding_boundary_report(actual_ratio,dense_ratio):
    """Only adjacent levels within eight FP32 ULPs of the same half-integer.

    This checks reduction-order differences; it never relaxes the check that
    stored codes equal round(actual_ratio). No parameter/v tolerance is changed.
    """
    if not bool(torch.isfinite(actual_ratio).all() & torch.isfinite(dense_ratio).all()):
        raise AssertionError('Nonfinite quantizer input')
    actual_level=actual_ratio.round().clamp(0,7)
    dense_level=dense_ratio.round().clamp(0,7)
    different=actual_level!=dense_level
    count=int(different.sum())
    if not count:return dict(count=0,max_distance_ulps=0.,examples=[])
    x,y=actual_ratio[different],dense_ratio[different]
    a,b=actual_level[different],dense_level[different]
    boundary=torch.minimum(a,b)+.5
    ulp=torch.nextafter(boundary,torch.full_like(boundary,float('inf')))-boundary
    distance=torch.maximum((x-boundary).abs(),(y-boundary).abs())/ulp
    if not bool(((a-b).abs().eq(1) & (distance<=8)).all()):
        raise AssertionError('Quantization mismatch outside adjacent eight-ULP boundary')
    coords=different.nonzero()[:8].tolist()
    return dict(count=count,max_distance_ulps=float(distance.max()),examples=[dict(index=pos,actual_ratio=float(actual_ratio[tuple(pos)]),dense_ratio=float(dense_ratio[tuple(pos)])) for pos in coords])

def verify_stored_history(state,current):
    """Independent dense statistics + exact codec check using observed inputs."""
    row=current.abs().mean(1);col=current.abs().mean(0)
    # The implementation reduces columns block by block, unlike the dense sum.
    # Check those statistics before examining any rounding boundary.
    for actual,expected in [(state.row,row),(state.col,col)]:
        torch.testing.assert_close(actual,expected,rtol=8*torch.finfo(torch.float32).eps,atol=0)
    scale=state.row[:,None]*state.col[None,:]/state.row.mean().clamp_min(1e-30)
    ratio=current.abs()/scale.clamp_min(1e-30)*(7/3)
    # Independent integer unpacking, not the production pack/unpack helpers.
    codes=torch.stack((state.codes%16,state.codes//16),dim=-1).reshape(state.rows,-1)[:,:state.cols]
    expected_codes=(ratio.round().clamp(0,7)+8*(current>=0)).to(torch.uint8)
    if not torch.equal(codes,expected_codes):raise AssertionError('Stored code violates actual nearest-level/sign rule')
    dense_scale=row[:,None]*col[None,:]/row.mean().clamp_min(1e-30)
    report=rounding_boundary_report(ratio,current.abs()/dense_scale.clamp_min(1e-30)*(7/3))
    decoded=scale*torch.where(current>=0,1.,-1.)*ratio.round().clamp(0,7)*(3/7)
    if not torch.equal(state.decode(0,state.rows),decoded):raise AssertionError('Decoded history differs from independently verified codes')
    return decoded,report

def capture_codec_inputs(opt):
    """Test-only full matrices; no capture hook is installed during training."""
    captured={}
    for k,state in enumerate(opt.m):
        if not getattr(state,'compressed',False):continue
        captured[k]=torch.empty((state.rows,state.cols),device=opt.device,dtype=torch.float32)
        def record(pending,current,row,diagnostic,original=state.second,target=captured[k]):
            target[row:row+current.shape[0]].copy_(current)
            return original(pending,current,row,diagnostic)
        state.second=record
    return captured

class DenseOracle:
    def __init__(self,named,chunk):
        self.named=named
        self.base=reference.R33(named,'rc',q=8,chunk=chunk,no_history='attention_embedding',drop_masked_m=True)
        self.m=[torch.zeros_like(p,dtype=torch.float32) for n,p in named]
        self.v=[tuple(t.clone() for t in v) if isinstance(v,tuple) else v.clone() for v in self.base.v]
    @torch.no_grad()
    def update(self,seeds,coeff,lr):
        gradients=[torch.zeros_like(p,dtype=torch.float32) for n,p in self.named]
        for k,start,p,zs in self.base.blocks(seeds):
            for c,z in zip(coeff,zs):gradients[k].view(-1)[start:start+p.numel()].add_(z,alpha=c)
        error=norm=0.
        self.last_a=[]
        for k,((name,p),g) in enumerate(zip(self.named,gradients)):
            a=g if self.base.no_history_mask[k] else self.m[k]*.9+g*.1
            self.last_a.append(a)
            v=self.v[k];square=a.square()
            if isinstance(v,tuple):
                v[0].mul_(.99).add_(square.mean(1),alpha=.01)
                v[1].mul_(.99).add_(square.mean(0),alpha=.01)
                den=v[0][:,None]*v[1][None,:]/v[0].mean().clamp_min(1e-30)
            else:v.mul_(.99).add_(square,alpha=.01);den=v
            p.add_(a/(den+1e-8).sqrt(),alpha=-lr)
            if '.mlp.' in name:
                r=a.abs().mean(1);c=a.abs().mean(0)
                scale=r[:,None]*c[None,:]/r.mean().clamp_min(1e-30)
                level=torch.round(a.abs()/scale.clamp_min(1e-30)*(7/3)).clamp(0,7)
                self.m[k]=torch.where(a>=0,1.,-1.)*scale*level*(3/7)
                error+=float((self.m[k].double()-a.double()).square().sum());norm+=float(a.double().square().sum())
            else:self.m[k]=a.clone()
        return (error/max(norm,1e-30))**.5

class RC4Tests(unittest.TestCase):
    def setUp(self):reference.DEVICE=DEVICE
    def test_all_codes_roundtrip_odd_width(self):
        for cols in [1,7,8,13,16,33]:
            codes=(torch.arange(3*cols,device=DEVICE).reshape(3,cols)%16).to(torch.uint8)
            packed=pack_codes(codes)
            self.assertEqual(packed.shape,(3,(cols+1)//2));self.assertEqual(packed.dtype,torch.uint8)
            self.assertTrue(torch.equal(unpack_codes(packed,cols),codes))
            if cols%2:self.assertEqual(int(packed[:,-1].bitwise_right_shift(4).sum()),0)

    def test_levels_clipping_sign_zero_and_sample_decode(self):
        a=torch.tensor([[-5.,-3.,-1.,0.,1.,3.,5.]],device=DEVICE)
        scale=torch.ones_like(a);code=encode(a,scale)
        expected=torch.tensor([[-3.,-3.,-6/7,0.,6/7,3.,3.]],device=DEVICE)
        torch.testing.assert_close(reconstruct(scale,code),expected)
        state=SignedRC4('x',a);pending=state.begin(1)
        state.accumulate(pending,a,0);state.finish_first(pending);state.second(pending,a,0,True);state.commit(pending)
        rows=torch.zeros(a.numel(),device=DEVICE,dtype=torch.long);cols=torch.arange(a.numel(),device=DEVICE)
        self.assertTrue(torch.equal(state.decode_samples(rows,cols),state.decode(0,1).flatten()))
        zero=SignedRC4('zero',a)
        self.assertEqual(float(zero.decode(0,1).abs().sum()),0.)
        opt=reference.optimizer(reference.parameters(),'signed_rc4');opt.update(list(range(8)),[0.]*8,1e-5)
        self.assertEqual(opt.compression_diagnostics['relative_frobenius_error'],0.)

    def test_multistep_dense_oracle_m_v_parameters_and_error(self):
        for dtype in [torch.float16,torch.float32]:
            for chunk in [127,1024]:
                pp=reference.parameters(dtype);qq=reference.parameters(dtype)
                opt=reference.optimizer(pp,'signed_rc4',chunk);oracle=DenseOracle(qq,chunk);capacity=opt.state_bytes()
                for step in range(5):
                    seeds=[step*113+j+3 for j in range(8)];coeff=[.13,-.2,.05,-.07,.11,.03,-.17,.06]
                    opt.update(seeds,coeff,1e-5);error=oracle.update(seeds,coeff,1e-5)
                    for k,((_,p),(_,ref)) in enumerate(zip(pp,qq)):
                        torch.testing.assert_close(p,ref,rtol=2e-3 if dtype==torch.float16 else 1e-5,atol=1e-4 if dtype==torch.float16 else 2e-7)
                        for a,b in zip(opt.v[k] if isinstance(opt.v[k],tuple) else (opt.v[k],),oracle.v[k] if isinstance(oracle.v[k],tuple) else (oracle.v[k],)):
                            torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-8)
                        state=opt.m[k]
                        if getattr(state,'compressed',False):torch.testing.assert_close(state.decode(0,state.rows),oracle.m[k],rtol=2e-4,atol=2e-6)
                        elif state is not None:torch.testing.assert_close(state,oracle.m[k])
                    if opt.last_diagnostics:self.assertAlmostEqual(error,opt.compression_diagnostics['relative_frobenius_error'],places=4)
                    self.assertEqual(opt.state_bytes(),capacity)

    def test_first_update_matches_original_exactly(self):
        a=reference.optimizer(reference.parameters(),'signed_rc');b=reference.optimizer(reference.parameters(),'signed_rc4')
        da=a.update(list(range(8)),[.03*j-.1 for j in range(8)],1e-5)
        db=b.update(list(range(8)),[.03*j-.1 for j in range(8)],1e-5)
        self.assertEqual(da,db)
        for (_,x),(_,y) in zip(a.named,b.named):self.assertTrue(torch.equal(x,y))
        for va,vb in zip(a.v,b.v):
            for x,y in zip(va if isinstance(va,tuple) else (va,),vb if isinstance(vb,tuple) else (vb,)):self.assertTrue(torch.equal(x,y))

    def test_observer_exact_state_parity_shadow_rng_and_direct_metrics(self):
        test_observer.DEVICE=DEVICE;original=reference.optimizer
        def create(named,mode,chunk=127,rank=16):return original(named,'signed_rc4',chunk,rank)
        with patch.object(reference,'optimizer',create):
            check=test_observer.ObserverTests()
            check.test_exact_parity_shadow_and_rng()
            check.test_sampling_metrics_match_direct_dense_samples()

    def test_bounded_allocations_and_exact_state_accounting(self):
        pp=reference.parameters(rows=257,cols=128);weight_storage={p.untyped_storage().data_ptr() for n,p in pp}
        class Audit(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                result=func(*args,**(kwargs or {}))
                def visit(value):
                    if torch.is_tensor(value) and value.dtype in [torch.float16,torch.float32,torch.float64] and value.numel()>=257*128 and value.untyped_storage().data_ptr() not in weight_storage:
                        raise AssertionError('Full FFN floating allocation: '+str(func))
                    if isinstance(value,(tuple,list)):
                        for v in value:visit(v)
                visit(result);return result
        with Audit():
            opt=reference.optimizer(pp,'signed_rc4',4096)
            opt.update(list(range(8)),[.1]*8,1e-5);opt.update(list(range(8,16)),[-.1]*8,1e-5)
        expected=0
        for (name,p),m in zip(pp,opt.m):
            if '.mlp.' in name:
                r,c=p.shape;expected+=r*((c+1)//2)+4*(r+c)
                self.assertFalse(hasattr(m,'signs'))
                self.assertEqual(set(vars(m)),{'rows','cols','row','col','codes'})
            elif m is not None:expected+=p.numel()*4
        self.assertEqual(opt.m_bytes(),expected)
        self.assertTrue(all(t.device==pp[0][1].device for t in opt.m_tensors()))

    def test_forward_count_and_exception_restoration(self):
        original=reference.optimizer
        def create(named,mode,chunk=127,rank=16):return original(named,'signed_rc4',chunk,rank)
        with patch.object(reference,'optimizer',create):reference.R34Tests().test_full_forward_count_and_exception_restoration()

    def test_production_chunk_rng_boundary(self):
        rows,cols,chunk=(2305,2048,4194304) if DEVICE.startswith('cuda') or FORCE_PRODUCTION_SHAPE else (137,32,2048)
        pp=reference.parameters(rows=rows,cols=cols)[:2]
        qq=[(n,torch.nn.Parameter(p.clone())) for n,p in pp]
        opt=reference.optimizer(pp,'signed_rc4',chunk);oracle=DenseOracle(qq,chunk)
        captured=capture_codec_inputs(opt);reports=[]
        for step in range(2):
            seeds=list(range(step*8,step*8+8));coeff=[.1*j-.23 for j in range(8)]
            opt.update(seeds,coeff,1e-5);oracle.update(seeds,coeff,1e-5)
            for (_,a),(_,b) in zip(pp,qq):torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-6)
            for va,vb in zip(opt.v,oracle.v):
                for a,b in zip(va,vb):torch.testing.assert_close(a,b,rtol=1e-4,atol=3e-7)
            for k,current in captured.items():
                torch.testing.assert_close(current,oracle.last_a[k],rtol=3e-5,atol=2e-8)
                verified,report=verify_stored_history(opt.m[k],current)
                reports.append(dict(step=step+1,name=opt.named[k][0],**report))
                # Both next-step recurrences start from the same *verified*
                # discrete history. They keep independent parameters and v.
                # Otherwise a legitimate rounding-boundary split propagates
                # into different algorithms/trajectories on the following step.
                oracle.m[k]=verified
        print('RC4_BOUNDARY_AUDIT '+json.dumps(dict(shape=[rows,cols],parameter_atol=2e-6,v_atol=3e-7,history_policy='independently verified quantized state shared before next step',rows=reports)),flush=True)

    def test_boundary_policy_and_exact_tie_rule(self):
        center=torch.arange(7,device=DEVICE,dtype=torch.float32)+.5
        down=torch.nextafter(center,torch.full_like(center,-float('inf')))
        up=torch.nextafter(center,torch.full_like(center,float('inf')))
        report=rounding_boundary_report(down,up)
        self.assertEqual(report['count'],7);self.assertEqual(report['max_distance_ulps'],1.)
        # Existing nearest-even behavior is unchanged at exact half-integers.
        self.assertTrue(torch.equal(center.round(),torch.tensor([0.,2.,2.,4.,4.,6.,6.],device=DEVICE)))
        self.assertEqual(rounding_boundary_report(center,center)['count'],0)
        for a,b in [(5.4,5.6),(3.5,5.5),(float('nan'),5.5)]:
            with self.assertRaises(AssertionError):rounding_boundary_report(torch.tensor([a],device=DEVICE),torch.tensor([b],device=DEVICE))
        ulp=torch.nextafter(center,torch.full_like(center,float('inf')))-center
        with self.assertRaises(AssertionError):rounding_boundary_report(center-9*ulp,center+ulp)

    def test_codec_gate_rejects_wrong_magnitude_sign_and_statistics(self):
        opt=reference.optimizer(reference.parameters(),'signed_rc4');captured=capture_codec_inputs(opt)
        opt.update(list(range(8)),[.1]*8,1e-5)
        k=next(iter(captured));state=opt.m[k];current=captured[k]
        verify_stored_history(state,current)
        for mask in [1,8]:
            before=state.codes[0,0].clone();state.codes[0,0].bitwise_xor_(mask)
            with self.assertRaises(AssertionError):verify_stored_history(state,current)
            state.codes[0,0].copy_(before)
        state.col.mul_(1.01)
        with self.assertRaises(AssertionError):verify_stored_history(state,current)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-cuda',action='store_true');p.add_argument('--production-shape',action='store_true');a,rest=p.parse_known_args()
    FORCE_PRODUCTION_SHAPE=a.production_shape
    if a.require_cuda:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        DEVICE='cuda:0'
    torch.set_num_threads(4);unittest.main(argv=[sys.argv[0],*rest])
