"""Choose original/optimized class; hash end states outside measured training."""
import argparse,hashlib,importlib,json,runpy,sys,time
from pathlib import Path
import torch

CHANGES=['native_inplace_direction_affine','reuse_second_pass_generators','skip_integer_isfinite']

def tensor_digest(t):
    if not t.is_contiguous():raise ValueError('Digest requires contiguous state')
    digest=hashlib.sha256();flat=t.detach().view(-1)
    # A bounded host copy for validation only. No GPU clone, saved history or
    # CPU optimizer computation. This runs after measured training/evaluation.
    for start in range(0,flat.numel(),4194304):
        digest.update(flat[start:start+4194304].view(torch.uint8).cpu().numpy().tobytes())
    return dict(shape=list(t.shape),dtype=str(t.dtype),numel=t.numel(),bytes=t.numel()*t.element_size(),sha256=digest.hexdigest())

def fingerprint(opt,path):
    start=time.perf_counter();items=[]
    for (name,p),m,v in zip(opt.named,opt.m,opt.v):
        items.append(dict(key='p/'+name,**tensor_digest(p)))
        mt=() if m is None else (m.tensors() if getattr(m,'compressed',False) else (m,))
        for i,t in enumerate(mt):items.append(dict(key=f'm/{name}/{i}',**tensor_digest(t)))
        for i,t in enumerate(v if isinstance(v,tuple) else (v,)):items.append(dict(key=f'v/{name}/{i}',**tensor_digest(t)))
    value=dict(scope='Parameters after final evaluation and final optimizer states; validation copies only, outside step and training timers.',updates=opt.updates,items=items,total_bytes=sum(x['bytes'] for x in items),seconds=time.perf_counter()-start,cpu_optimizer_offload=False)
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');return value

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--r37-mode',required=True,choices=['original','optimized']);p.add_argument('--r37-fingerprint',required=True,type=Path)
    a,remaining=p.parse_known_args()
    module=importlib.import_module('compressed_fzoo_fast' if a.r37_mode=='optimized' else 'compressed_fzoo')
    sys.modules['compressed_fzoo']=module
    original=module.CompressedFZOO.metadata;captured={}
    def metadata(self):
        if self.compression!='signed_rc4':raise ValueError('R37 only benchmarks the approved four-bit candidate')
        captured['optimizer']=self
        return dict(original(self),r37_speed_mode=a.r37_mode,r37_changes=CHANGES if a.r37_mode=='optimized' else [])
    module.CompressedFZOO.metadata=metadata
    sys.argv=[str(Path(__file__).with_name('diag_entry.py'))]+remaining
    runpy.run_path(sys.argv[0],run_name='__main__')
    fingerprint(captured['optimizer'],a.r37_fingerprint)
if __name__=='__main__':main()
