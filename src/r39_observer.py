"""Read-only sampled FFN observer. All shadow histories stay on the GPU."""
import hashlib,json,math,random
from pathlib import Path
import torch

def stats(x,y):
    x=x.double();y=y.double()
    return dict(n=x.numel(),xx=float(x.square().sum()),yy=float(y.square().sum()),
                dot=float((x*y).sum()),error=float((x-y).square().sum()),
                sign_disagree=int(((x>=0)!=(y>=0)).sum()))

def measures(s):
    a,b=s['xx'],s['yy']
    return dict(samples=s['n'],relative_error=math.sqrt(s['error']/max(b,1e-30)),
        cosine=s['dot']/max(math.sqrt(a*b),1e-30) if a or b else 1.,
        norm_ratio=math.sqrt(a/max(b,1e-30)),sign_disagreement=s['sign_disagree']/s['n'])

class Sample:
    def __init__(self,state,name,shape,count,interval):
        self.state=state;self.name=name;self.shape=shape;self.interval=interval
        n=math.prod(shape);count=min(count,n)
        seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')
        # A local RNG avoids stride aliasing onto one column and never
        # consumes Python's global RNG or the model/direction generators.
        self.indices=sorted(random.Random(seed).sample(range(n),count))
        device=state.row.device
        self.rows=torch.tensor([i//shape[1] for i in self.indices],device=device,dtype=torch.long)
        self.cols=torch.tensor([i%shape[1] for i in self.indices],device=device,dtype=torch.long)
        self.shadow=torch.zeros(count,device=device,dtype=torch.float32)
        self.current=torch.zeros_like(self.shadow)
        self.maps={};self.step=0;self.first=False;self.seen=0
        begin,combine,finish=state.begin,state.combine,state.finish_first
        def observed_begin(step):
            self.step=step;self.first=True;self.seen=0
            return begin(step)
        def observed_combine(g,row,beta):
            value=combine(g,row,beta)
            if self.first:
                key=(row,g.shape[0])
                if key not in self.maps:
                    lo=row*shape[1];hi=lo+g.numel()
                    selected=[(j,i-lo) for j,i in enumerate(self.indices) if lo<=i<hi]
                    self.maps[key]=(torch.tensor([j for j,i in selected],device=device,dtype=torch.long),
                                    torch.tensor([i for j,i in selected],device=device,dtype=torch.long))
                slots,local=self.maps[key]
                if slots.numel():
                    exact=self.shadow.index_select(0,slots).mul_(beta).add_(g.reshape(-1).index_select(0,local),alpha=1-beta)
                    self.shadow.index_copy_(0,slots,exact)
                    if self.due():self.current.index_copy_(0,slots,value.reshape(-1).index_select(0,local))
                    self.seen+=slots.numel()
            return value
        def observed_finish(pending):
            if self.seen!=count:raise RuntimeError('Sample coverage mismatch: '+name)
            result=finish(pending);self.first=False
            return result
        state.begin=observed_begin;state.combine=observed_combine;state.finish_first=observed_finish

    def due(self):return (self.step-1)%self.interval==0

    def tensors(self):
        return [self.rows,self.cols,self.shadow,self.current]+[t for pair in self.maps.values() for t in pair]

    def collect(self,v):
        s=self.state
        if hasattr(s,'decode_samples'):
            decoded=s.decode_samples(self.rows,self.cols)
        else:
            bits=s.signs[self.rows,self.cols//8].bitwise_right_shift(self.cols%8).bitwise_and(1).float()
            decoded=s.row[self.rows]*s.col[self.cols]/s.row.mean().clamp_min(1e-30)*(bits*2-1)
        denom=(v[0][self.rows]*v[1][self.cols]/v[0].mean().clamp_min(1e-30)+1e-8).sqrt()
        return dict(name=self.name,shape=list(self.shape),layer=int(self.name.split('.layers.')[1].split('.')[0]),
            projection=self.name.split('.mlp.')[1].split('.')[0] if '.mlp.' in self.name else self.name.split('.')[-2],sample_count=len(self.indices),
            raw=dict(codec=stats(decoded,self.current),moment=stats(self.current,self.shadow),
                     stored_history=stats(decoded,self.shadow),
                     update_same_v=stats(self.current/denom,self.shadow/denom)))

class Observer:
    def __init__(self,opt,path,samples=256,interval=50):
        if samples<1 or interval<1:raise ValueError('Positive sample count and interval required')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.opt=opt;self.parts={}
        for k,((name,p),state) in enumerate(zip(opt.named,opt.m)):
            if '.mlp.' in name and getattr(state,'mode',None) in ('signed_rc','signed_rc4'):self.parts[k]=Sample(state,name,tuple(p.shape),samples,interval)
        if not self.parts:raise ValueError('Signed FFN states required')
        self.write(dict(kind='observer_init',samples_per_module=samples,interval=interval,
            modules=[dict(name=s.name,shape=s.shape,indices=s.indices) for s in self.parts.values()],
            scope='Fixed sampled coordinates; exact EMA of gradients on THIS compressed trajectory, not a separate uncompressed-model trajectory.',
            update_reference='Both updates use actual compressed-run row-column v; numerator-only comparison. No exact uncompressed v is stored.',
            cpu_history_offload=False))

    def write(self,value):
        with self.path.open('a',encoding='utf8') as h:h.write(json.dumps(value,allow_nan=False)+'\n')

    def finish(self):
        first=next(iter(self.parts.values()))
        if not first.due():return
        modules=[]
        for k,s in self.parts.items():
            row=s.collect(self.opt.v[k]);row['metrics']={k:measures(v) for k,v in row['raw'].items()};modules.append(row)
        total={}
        for key in modules[0]['raw']:
            raw={f:sum(m['raw'][key][f] for m in modules) for f in modules[0]['raw'][key]}
            total[key]=measures(raw)
        tensors=[t for s in self.parts.values() for t in s.tensors()]
        persistent=sum(t.numel()*t.element_size() for t in tensors)
        self.write(dict(kind='observer',step=self.opt.updates,modules=modules,aggregate=total,
                        observer_persistent_bytes=persistent,algorithm_state_bytes=self.opt.state_bytes()))

def attach(opt,path,samples=256,interval=50):
    if hasattr(opt,'_r35_observer'):raise ValueError('Observer already attached')
    observer=Observer(opt,path,samples,interval);original=opt.update
    def update(*args,**kwargs):
        result=original(*args,**kwargs);observer.finish();return result
    opt.update=update;opt._r35_observer=observer
    return observer

def install(path,samples=256,interval=50):
    from compressed_fzoo import CompressedFZOO
    original=CompressedFZOO.__init__
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs);attach(self,path,samples,interval)
    CompressedFZOO.__init__=initialize
