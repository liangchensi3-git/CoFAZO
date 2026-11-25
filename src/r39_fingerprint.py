"""Bounded post-training checksums, including stateless FZOO controls."""
import json,time
from pathlib import Path
from speed_entry import tensor_digest

def fingerprint(opt,path):
    start=time.perf_counter();items=[]
    for (name,p),m,v in zip(opt.named,opt.m,opt.v):
        items.append(dict(key='p/'+name,**tensor_digest(p)))
        mt=() if m is None else (m.tensors() if getattr(m,'compressed',False) else (m,))
        vt=() if v is None else (v if isinstance(v,tuple) else (v,))
        for i,t in enumerate(mt):items.append(dict(key=f'm/{name}/{i}',**tensor_digest(t)))
        for i,t in enumerate(vt):items.append(dict(key=f'v/{name}/{i}',**tensor_digest(t)))
    value=dict(scope='Parameters after final evaluation, final optimizer states; no training resume checkpoint.',updates=opt.updates,items=items,total_bytes=sum(x['bytes'] for x in items),seconds=time.perf_counter()-start,cpu_optimizer_offload=False)
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');return value
