"""Isolated Qwen loading; OPT retains its historical loader unchanged."""
import json
from pathlib import Path


def model_family(path):
    return json.loads((Path(path)/'config.json').read_text(encoding='utf-8'))['model_type']


def install_loader(run):
    import sys
    model_arg=sys.argv[sys.argv.index('--model_name')+1]
    if model_family(model_arg)=='qwen3':
        import transformers.utils as utilities
        sys.argv[:]=['--eval_strategy' if x=='--evaluation_strategy' else x for x in sys.argv]
        # The old TPU capability name was renamed; no TPU backend is used.
        utilities.is_torch_tpu_available=lambda check_device=True: utilities.is_torch_xla_available()
    original = run.Framework.load_model
    def load(self):
        family = model_family(self.args.model_name)
        if family == 'opt':
            return original(self)
        if family != 'qwen3':
            raise ValueError('R28 supports local OPT or Qwen3 Base models only')
        install_trainer_compat()
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        config=AutoConfig.from_pretrained(self.args.model_name,local_files_only=True,trust_remote_code=False)
        config.use_cache=False
        model=AutoModelForCausalLM.from_pretrained(self.args.model_name,config=config,
            torch_dtype=torch.float16,local_files_only=True,trust_remote_code=False,
            device_map={'':0} if torch.cuda.is_available() else None)
        tokenizer=AutoTokenizer.from_pretrained(self.args.model_name,use_fast=True,
            local_files_only=True,trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None: raise ValueError('No pad or EOS token in local tokenizer')
            tokenizer.pad_token=tokenizer.eos_token
        model.config.pad_token_id=tokenizer.pad_token_id
        model.eval()
        return model,tokenizer
    run.Framework.load_model=load


def install_trainer_compat():
    """Bridge the vendored 4.28 loop to pinned 4.51 callbacks, Qwen only."""
    import time
    import transformers
    from transformers import Trainer, TrainerCallback
    from mezotrainer import OurTrainer
    if transformers.__version__ != '4.51.3':
        raise RuntimeError('Qwen3 requires the bundled Transformers4.51.3')
    if getattr(OurTrainer,'_r28_qwen_compat',False): return
    original_init=OurTrainer.__init__
    class Intervals(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            for name in ('logging_steps','eval_steps','save_steps'):
                value=getattr(args,name)
                if value is None or value<1:
                    raise ValueError('R28 requires absolute callback intervals')
                setattr(state,name,int(value))
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.add_callback(Intervals())
        self._r28_start_time=time.time()
    def maybe(self,tr_loss,model,trial,epoch,ignore_keys_for_eval):
        return Trainer._maybe_log_save_evaluate(self,tr_loss,None,model,trial,epoch,
                                                ignore_keys_for_eval,self._r28_start_time)
    OurTrainer.__init__=init
    OurTrainer._maybe_log_save_evaluate=maybe
    OurTrainer._r28_qwen_compat=True
