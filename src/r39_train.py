"""Full-parameter streaming FZOO runner with q+1 counted objectives."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch
from single_pass import SinglePassFZOO as FullFZOO
from r39_schedule import effective_lr, ARMS, BOOST


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--r39-arm',choices=ARMS,required=True)
    parser.add_argument('--r22-kind',choices=('fzoo','full','rc','momentum'),required=True)
    parser.add_argument('--q',type=int,default=8)
    parser.add_argument('--r32-no-history',choices=('none','ffn','attention','embedding','attention_embedding'),default='none')
    parser.add_argument('--r33-drop-masked-m',action='store_true')
    parser.add_argument('--r34-m-compression',choices=('none','signed_rc','signed_rc4','lowrank'),default='none')
    parser.add_argument('--r34-rank',type=int,default=16)
    parser.add_argument('--r22-diag-interval',type=int,default=100)
    parser.add_argument('--r22-artifacts',type=Path,required=True)
    parser.add_argument('--r22-chunk',type=int,default=1048576)
    parser.add_argument('--m-horizon',type=int,choices=(0,8,16),default=0)
    parser.add_argument('--r28-drop-after',type=int,default=0)
    parser.add_argument('--r28-drop-factor',type=float,default=.25)
    opts,rest=parser.parse_known_args(); sys.argv=[sys.argv[0],*rest]
    peak=BOOST[opts.r39_arm]; captured={}
    if opts.m_horizon != 0:
        raise ValueError('R28 uses stored m only')
    effective_lr(1.,1,opts.r28_drop_after,opts.r28_drop_factor)
    torch.set_num_threads(4)
    out=opts.r22_artifacts.resolve(); out.mkdir(parents=True,exist_ok=True)
    trace=out/'trace.jsonl'
    if trace.exists(): raise FileExistsError(trace)
    def emit(kind,**values):
        with trace.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'kind':kind,**values},allow_nan=False)+'\n')
    def sync():
        if torch.cuda.is_available(): torch.cuda.synchronize()
    import run
    from qwen_adapter import install_loader
    install_loader(run)
    from mezotrainer import OurTrainer
    original_train=run.Framework.train

    @torch.no_grad()
    def step(self,model,inputs):
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if self.args.gradient_accumulation_steps!=1 or self.args.lora or len(named)!=len(list(model.parameters())):
            raise ValueError('Requires full-parameter training and accumulation=1')
        if not hasattr(self,'r22_optimizer'):
            expected='cuda:0' if torch.cuda.is_available() else 'cpu'
            devices=sorted({str(p.device) for p in model.parameters()}|{str(b.device) for b in model.buffers()})
            if devices!=[expected] or any(p.dtype!=torch.float16 for p in model.parameters()):
                raise ValueError('Entire model must be FP16 on one device; got '+str(devices))
            if opts.r32_no_history!='none' and (model.config.model_type!='qwen3' or opts.r22_kind!='rc'):
                raise ValueError('R32 ablation requires Qwen3 rc')
            model.config.use_cache=False
            optimizer_class=FullFZOO;compression_options={}
            if opts.r34_m_compression!='none':
                from compressed_fzoo_fast import CompressedFZOO
                from r39_all_m import AllMatrixRC4
                optimizer_class=AllMatrixRC4 if opts.r39_arm=='all_rc4' else CompressedFZOO
                compression_options=dict(m_compression=opts.r34_m_compression,rank=opts.r34_rank)
            self.r22_optimizer=optimizer_class(named,opts.r22_kind,q=opts.q,mu=self.args.zo_eps,beta1=self.args.zo_beta1,beta2=self.args.zo_beta2,zeta=self.args.zo_adam_eps,diagnostics_interval=opts.r22_diag_interval,chunk=opts.r22_chunk,no_history=opts.r32_no_history,drop_masked_m=opts.r33_drop_masked_m,**compression_options)
            captured['optimizer']=self.r22_optimizer
            if compression_options:
                from r39_observer import attach
                attach(self.r22_optimizer,out/'observer.jsonl',samples=256,interval=50)
            self.r28_training_seconds=0.
            self.r22_rng=torch.Generator().manual_seed(self.args.seed+140230)
            self.r22_zero_streak=0
            self.r22_model_calls=0
            def counted(*args): self.r22_model_calls+=1
            self.r22_hook=model.register_forward_pre_hook(counted)
            masked=[(n,p) for (n,p),take in zip(named,self.r22_optimizer.no_history_mask) if take]
            metadata=self.r22_optimizer.metadata() if compression_options else dict(m_compression='none',compression_rank=None)
            m_bytes=self.r22_optimizer.m_bytes() if compression_options else sum(m.numel()*m.element_size() for m in self.r22_optimizer.m if m is not None)
            emit('init',r39_arm=opts.r39_arm,early_boost=peak,boost_until=3000,
                 observer_scope='ffn_only' if compression_options else 'off',
                 no_history=opts.r32_no_history,no_history_names=[n for n,p in masked],
                 no_history_numel=sum(p.numel() for n,p in masked),no_history_m_bytes=sum(4*p.numel() for n,p in masked),
                 m_storage_retained=not opts.r33_drop_masked_m,drop_masked_m=opts.r33_drop_masked_m,
                 dropped_m_bytes=sum(4*p.numel() for n,p in masked) if opts.r33_drop_masked_m else 0,
                 gradient_reconstruction=('current_step_all_matrices' if opts.r39_arm=='all_rc4' else 'current_step_selected_and_ffn') if compression_options else ('current_step_selected_tensors' if opts.r33_drop_masked_m else 'none'),
                 drop_after=opts.r28_drop_after,drop_factor=opts.r28_drop_factor,model_type=model.config.model_type,
                 m_horizon=opts.m_horizon,m_bytes=m_bytes,mu=self.r22_optimizer.mu,beta1=self.r22_optimizer.b1,
                 beta2=self.r22_optimizer.b2,zeta=self.r22_optimizer.zeta,optimizer=opts.r22_kind,chunk_elements=self.r22_optimizer.chunk,trainable_numel=self.r22_optimizer.d,trainable_tensors=len(named),
                 implementation='fp16_sparse_single_pass',direction_cache_bytes=0,diagnostics_interval=opts.r22_diag_interval,
                 moment_dtype='torch.float32',master_weights=False,
                 trainable_dtypes=['torch.float16'],devices=devices,estimator='rademacher_q_forward_std_normalized',q=opts.q,state_bytes=self.r22_optimizer.state_bytes(),
                 zeta_inside_sqrt=True,model_numel=sum(p.numel() for p in model.parameters()),**metadata)
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        model.eval()
        inputs=self._prepare_inputs(inputs)
        digest=hashlib.sha256()
        for key in ('input_ids','labels'):
            v=inputs.get(key)
            if torch.is_tensor(v): digest.update(key.encode()+v.detach().cpu().contiguous().numpy().tobytes())
        seeds=torch.randint(0,2**31-1,(opts.q,),generator=self.r22_rng).tolist()
        self.r22_model_calls=0
        times=[]
        def closure():
            sync(); start=time.perf_counter()
            with torch.inference_mode(),self.compute_loss_context_manager():
                loss=self.compute_loss(model,inputs)
            sync(); times.append(time.perf_counter()-start)
            if self.state.global_step==0 and len(times)==1:
                emit('forward_reference',allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                     reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
                     note='FP16 full model, FP32 moments, no master weights; not pure inference memory')
            return loss
        sync(); start=time.perf_counter()
        base_lr=self._get_learning_rate()
        used_lr=effective_lr(base_lr,self.state.global_step+1,opts.r28_drop_after,opts.r28_drop_factor,peak=peak)
        evidence=self.r22_optimizer.step(closure,seeds,used_lr)
        sync(); elapsed=time.perf_counter()-start
        self.r28_training_seconds+=elapsed
        if self.state.global_step==0:self.r22_initial_loss=evidence['loss']
        if self.r22_model_calls!=opts.q+1: raise RuntimeError('Expected q+1 model forwards per update')
        self.r22_zero_streak=self.r22_zero_streak+1 if evidence['loss_difference']==0 else 0
        if self.r22_zero_streak>=20: raise FloatingPointError('20 consecutive zero loss differences; no useful gradient evidence')
        emit('step',step=self.state.global_step+1,**evidence,finite=True,seeds=seeds,batch_hash=digest.hexdigest(),
             lr=used_lr,base_lr=base_lr,seconds=elapsed,forward_seconds=times[0],all_forward_seconds=sum(times),
             forward_calls=self.r22_model_calls,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
             peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None)
        return next(model.parameters()).new_tensor(evidence['loss'])

    def evaluate(self,*args,**kwargs):
        start=time.perf_counter()
        self.model._r28_eval_counts=torch.zeros(2,device=next(self.model.parameters()).device,dtype=torch.int64)
        try:
            result=super(OurTrainer,self).evaluate(*args,**kwargs)
            correct,total=self.model._r28_eval_counts.tolist()
        finally:
            del self.model._r28_eval_counts
        if 'eval_loss' in result:
            if total<=0 or not 0<=correct<=total:
                raise ValueError('Missing development classification counts')
            result['eval_accuracy']=correct/total
            loss=float(result['eval_loss'])
            if not math.isfinite(loss): raise FloatingPointError('Nonfinite eval loss')
            if hasattr(self,'r22_initial_loss') and loss>max(2*self.r22_initial_loss,self.r22_initial_loss+1):
                raise FloatingPointError('Dev loss exceeds conservative divergence gate')
            if loss<self._best_eval_loss:
                self._best_eval_loss=loss; self._best_step=self.state.global_step
                self._best_state_dict={n:p.detach().to('cpu',copy=True) for n,p in self.model.named_parameters() if p.requires_grad}
            emit('eval',step=self.state.global_step,loss=loss,accuracy=correct/total,correct=correct,count=total,
                 seconds=time.perf_counter()-start,training_seconds=getattr(self,'r28_training_seconds',0.),
                 training_forward_calls=self.state.global_step*(opts.q+1),elapsed_seconds=time.perf_counter()-run_started)
        return result
    def load_best(self):
        if self._best_state_dict is None: return False
        self.model.load_state_dict(self._best_state_dict,strict=False); return True
    def train(self,train_samples,eval_samples):
        digest=lambda s:hashlib.sha256(json.dumps([x.data for x in s],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        emit('data',ntrain=len(train_samples),ndev=len(eval_samples),train_hash=digest(train_samples),dev_hash=digest(eval_samples))
        return original_train(self,train_samples,eval_samples)
    OurTrainer.zo_step=step
    OurTrainer.zo_update=lambda self,model:self.lr_scheduler.step()
    OurTrainer.evaluate=evaluate; OurTrainer.load_best_state=load_best
    run.Framework.train=train
    run_started=time.perf_counter(); run.main()
    emit('complete',wall_seconds=time.perf_counter()-run_started,
         peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
         peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None)
    from r39_fingerprint import fingerprint
    fingerprint(captured['optimizer'],out/'state_fingerprint.json')

if __name__=='__main__': main()
