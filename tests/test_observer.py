import argparse,json,random,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'src'))
import test_r34 as reference
from observer import attach,measures,stats
DEVICE='cpu'

class ObserverTests(unittest.TestCase):
    def test_exact_parity_shadow_and_rng(self):
        reference.DEVICE=DEVICE
        for dtype in [torch.float16,torch.float32]:
            for chunk in [127,1024]:
                with tempfile.TemporaryDirectory() as tmp:
                    a=reference.optimizer(reference.parameters(dtype),'signed_rc',chunk)
                    b=reference.optimizer(reference.parameters(dtype),'signed_rc',chunk)
                    rng_before=random.getstate()
                    observer=attach(b,Path(tmp)/'observer.jsonl',samples=17,interval=2)
                    self.assertEqual(rng_before,random.getstate())
                    shadows={k:torch.zeros_like(b.named[k][1],dtype=torch.float32) for k in observer.parts}
                    capacities=[]
                    for step in range(4):
                        seeds=[step*113+j+3 for j in range(8)];coeff=[.13,-.2,.05,-.07,.11,.03,-.17,.06]
                        for k,start,p,zs in a.blocks(seeds):
                            if k in shadows:
                                g=torch.zeros_like(p,dtype=torch.float32)
                                for c,z in zip(coeff,zs):g.add_(z,alpha=c)
                                shadows[k].reshape(-1)[start:start+p.numel()].mul_(.9).add_(g,alpha=.1)
                        before=torch.cuda.get_rng_state() if DEVICE.startswith('cuda') else torch.get_rng_state()
                        da=a.update(seeds,coeff,1e-5);db=b.update(seeds,coeff,1e-5)
                        after=torch.cuda.get_rng_state() if DEVICE.startswith('cuda') else torch.get_rng_state()
                        self.assertTrue(torch.equal(before,after));self.assertEqual(da,db)
                        self.assertEqual(a.compression_diagnostics,b.compression_diagnostics)
                        self.assertEqual(a.state_bytes(),b.state_bytes())
                        for (_,pa),(_,pb) in zip(a.named,b.named):self.assertTrue(torch.equal(pa,pb))
                        for va,vb in zip(a.v,b.v):
                            for x,y in zip(va if isinstance(va,tuple) else (va,),vb if isinstance(vb,tuple) else (vb,)):self.assertTrue(torch.equal(x,y))
                        for x,y in zip(a.m_tensors(),b.m_tensors()):self.assertTrue(torch.equal(x,y))
                        for k,s in observer.parts.items():
                            idx=torch.tensor(s.indices,device=DEVICE)
                            # Small gathered vectors and large contiguous CPU/CUDA
                            # kernels can differ by a few FP32 rounding bits.
                            # Actual optimizer states/parameters above remain bitwise equal.
                            torch.testing.assert_close(s.shadow,shadows[k].reshape(-1)[idx],rtol=2e-6,atol=2e-8)
                        capacities.append(sum(t.numel()*t.element_size() for s in observer.parts.values() for t in s.tensors()))
                    self.assertEqual(len(set(capacities)),1)
                    logs=[json.loads(x) for x in (Path(tmp)/'observer.jsonl').read_text().splitlines()]
                    self.assertEqual([r['step'] for r in logs[1:]],[1,3])
                    self.assertEqual(logs[1]['aggregate']['moment']['relative_error'],0.)
                    self.assertEqual(len(logs[1]['modules']),3)
                    for s in observer.parts.values():
                        self.assertEqual(len(s.indices),len(set(s.indices)))
                        self.assertGreater(len({i%s.shape[1] for i in s.indices}),1)
                        self.assertGreater(len({i//s.shape[1] for i in s.indices}),1)

    def test_sampling_metrics_match_direct_dense_samples(self):
        reference.DEVICE=DEVICE
        with tempfile.TemporaryDirectory() as tmp:
            opt=reference.optimizer(reference.parameters(),'signed_rc')
            obs=attach(opt,Path(tmp)/'obs.jsonl',samples=10000,interval=1)
            opt.update(list(range(8)),[.1]*8,1e-5)
            for k,s in obs.parts.items():
                row=s.collect(opt.v[k]);decoded=s.state.decode(0,s.shape[0])
                idx=torch.tensor(s.indices,device=DEVICE)
                direct=metrics=measures(stats(decoded.reshape(-1)[idx],s.current))
                got=measures(row['raw']['codec'])
                for key in direct:self.assertAlmostEqual(got[key],direct[key],places=6)

    def test_zero_state_is_finite(self):
        reference.DEVICE=DEVICE
        with tempfile.TemporaryDirectory() as tmp:
            opt=reference.optimizer(reference.parameters(),'signed_rc');attach(opt,Path(tmp)/'o.jsonl')
            opt.update(list(range(8)),[0.]*8,1e-5)
            rows=[json.loads(x) for x in (Path(tmp)/'o.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['aggregate']['moment']['relative_error'],0.)

    def test_invalid_and_duplicate_attachment(self):
        reference.DEVICE=DEVICE
        with tempfile.TemporaryDirectory() as tmp:
            opt=reference.optimizer(reference.parameters(),'signed_rc')
            with self.assertRaises(ValueError):attach(opt,Path(tmp)/'bad',samples=0)
            attach(opt,Path(tmp)/'ok')
            with self.assertRaises(ValueError):attach(opt,Path(tmp)/'twice')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-cuda',action='store_true');a,rest=p.parse_known_args()
    if a.require_cuda:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        DEVICE='cuda:0'
    torch.set_num_threads(4);unittest.main(argv=[sys.argv[0],*rest])
