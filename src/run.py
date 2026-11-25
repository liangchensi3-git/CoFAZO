import logging
import pdb
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import os
os.environ["WANDB_DISABLED"]="true"

import argparse
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification, TrainerCallback
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
# from trainer import OurTrainer
import random
# export HF_ENDPOINT=https://hf-mirror.com
from ParallelOPTForCausalLM import ParallelOPTForCausalLM
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    method: str = "FZOO" # FZOO, FZOO-AdaMM, FZOO-R-AdaZO, FZOO-Decoupled, MeZO, ZO-AdaMM, ZO-RMSProp, R-AdaZO, R-AdaZO-Official

    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samplesOurArgumentsrgs.deepspeed
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 
    """
    turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
    """
    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO
    # Adaptive ZO (paper defaults for R-AdaZO LLM fine-tuning: 0.9, 0.99)
    zo_beta1: float = 0.9
    zo_beta2: float = 0.99
    zo_adam_eps: float = 1e-8
    # R13 diagnostics. Defaults are the exact R12 FZOO estimator/update path.
    zo_direction: str = "rademacher"  # rademacher / gaussian
    zo_difference: str = "single"  # single-sided / central (two-sided)
    zo_sigma_normalize: int = 1  # 0 = raw directional derivative divided by zo_eps
    zo_adam_bias_correction: int = 0  # diagnostic only; official R-AdaZO uses 0
    zo_radazo_g2_warmup: int = 0  # first K steps use g^2 for v to test initial overshoot
    zo_radazo_g2_after: int = 0  # if positive, switch back to g^2 after this step
    zo_decoupled_direction: str = "m2"  # FZOO-Decoupled direction: m2 / g2
    zo_decoupled_norm: str = "m2"  # FZOO-Decoupled global L2 reference: m2 / g2
    zo_update_diag: int = 0  # stream g/m/v/update norms and batch hashes
    zo_trace_diag: int = 0  # R16 common l0/l+/l-/projected diagnostics; default off
    zo_stop_step: int = 0  # stop after N steps without shortening the LR schedule

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)
    # APPEND FOR LAZO
    Th : float = 0.5
    # loss batch
    parallel : int = 1
    # APPEND FOR ARS
    N : int = 8
    # K : float = 1.0
    thre : int = 0
    D : int = 1
    # FZOO-M: momentum coefficient beta on the normalized direction (0 = original FZOO)
    zo_momentum : float = 0.0
    # Diagnostics: every N steps log cos(ZO estimate, true grad) etc. (0 = off)
    zo_diag_interval : int = 0
    # FZOO-M: EMA smoothing of the per-step loss-std normalizer sigma_t (0 = off).
    # v = b2*v + (1-b2)*sigma_t^2 with bias correction; sigma_hat replaces sigma_t.
    zo_sigma_ema : float = 0.0
    # FZOO-M: early stop when dev eval_loss rebounds above best*(1+thresh)
    # for `zo_early_stop_patience` consecutive evals (thresh 0 = off).
    zo_early_stop_thresh : float = 0.0
    zo_early_stop_patience : int = 2
    # FZOO-M: keep an in-memory snapshot of the best-dev-loss model during
    # training; after the usual final-model eval, also report metrics of the
    # best checkpoint (keys prefixed bestckpt_). 0 = off.
    zo_best_ckpt : int = 0
    # FZOO-M: first-order oracle. 1 = skip ZO perturbations, compute the true
    # gradient with one backprop per step, and run plain SGD / SGD+momentum
    # under the SAME protocol (data, schedule, evals). zo_momentum is reused
    # as the EMA beta. Answers "does momentum help with true gradients here?"
    # and gives the upper bound that any ZO momentum approximation can aim at.
    fo_sgd : int = 0
    # FZOO-M: which first-order optimizer the FO oracle uses (fo_sgd=1 时生效):
    # "sgd" = 裸 SGD(EMA 动量由 zo_momentum 控制); "adam" = Adam(β1=0.9, β2=0.999);
    # "rmsprop" = 纯二阶矩(β1=0), 用于分离 Adam 收益中 m 与 v 的贡献(r7a)。
    fo_optimizer : str = "sgd"
    # FZOO-RC: 行/列分解二阶矩(Adafactor 式 v)的 EMA 系数 β2, 0 = 关闭。
    # 对 ZO 估计 g_hat 的行/列能量做去偏(减去已知的扰动噪声贡献 sum(alpha^2))
    # 后 EMA, 预条件方向 = g_hat / sqrt(v_hat), v_hat 归一到均值 1 保持 lr 可比。
    zo_rc_v : float = 0.0
    # FZOO-RC: warmup 步数, 期内 v 未积累、回退 base ZO 方向(β2=0.98 约需 50 步)。
    zo_rc_warmup : int = 50
    # FZOO-RC: vhat 均值归一后的地板, 预条件放大硬限 1/sqrt(vfloor) (默认 10x)。
    # r7b 事故修复: 去偏 clamp 使过半 vhat 为精确 0, 无地板时首步 fp16 溢出 NaN。
    zo_rc_vfloor : float = 0.01
    # FZOO-RC: 1 = 只缩不放(r10 v3): vhat 地板改为 1.0, 高 v 坐标降权、低 v 坐标原样,
    # 绝不放大无信号坐标(r9: 放大死坐标上的 ZO 纯噪声使 cos_pre 稳定低于 cos_ghat)。
    zo_rc_oneside : int = 0
    # 动量形式(r10b): "ema" = buf=β·buf+(1-β)·g(r5/r6 历史行为, lr 与无动量直接可比);
    # "heavyball" = buf=β·buf+g(经典/PyTorch 形式, 一致方向上有效步长放大 1/(1-β),
    # 注意 stated lr 的含义随之改变, 网格需按 1/(1-β) 折算)。FO/ZO 动量分支共用。
    momentum_style : str = "ema"
class DiagnosticStopCallback(TrainerCallback):
    """Stop a short diagnostic run without changing scheduler total steps."""

    def __init__(self, stop_step):
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            logger.info("Diagnostic stop reached at step %d; LR schedule total remains %d",
                        state.global_step, args.max_steps)
            control.should_training_stop = True
        return control



def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()
        self.trainer = None
        # self.model._custom_n = None        ##
        # self.model._custom_perturb = None  ##
        print("self.model:",type(self.model)) # <class 'transformers.models.opt.modeling_opt.OPTForCausalLM'>
        print("self.model:",self.model)
        # if "opt" in self.args.model_name.lower():
        #     from replace_moudle import replace_opt_linear#,replace_opt_attention 
        #     replace_opt_linear(self.model)
        #     # replace_opt_attention(self.model)
        # print("self.model:",type(self.model)) # <class 'transformers.models.opt.modeling_opt.OPTForCausalLM'>
        # print("self.model:",self.model)
        """
        self.model: OPTForCausalLM(
        (model): OPTModel(
            (decoder): OPTDecoder(
            (embed_tokens): Embedding(50272, 768, padding_idx=1)
            (embed_positions): OPTLearnedPositionalEmbedding(2050, 768)
            (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (layers): ModuleList(
                (0-11): 12 x OPTDecoderLayer(
                (self_attn): OPTAttention(
                    (k_proj): Linear(in_features=768, out_features=768, bias=True)
                    (v_proj): Linear(in_features=768, out_features=768, bias=True)
                    (q_proj): Linear(in_features=768, out_features=768, bias=True)
                    (out_proj): Linear(in_features=768, out_features=768, bias=True)
                )
                (activation_fn): ReLU()
                (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (fc1): Linear(in_features=768, out_features=3072, bias=True)
                (fc2): Linear(in_features=3072, out_features=768, bias=True)
                (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                )
            )
            )
        )
        (lm_head): Linear(in_features=768, out_features=50272, bias=False)
        )

        """


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            has_cuda = torch.cuda.is_available()
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3) if has_cuda else 0
            config = AutoConfig.from_pretrained(self.args.model_name)
            # print("self.args.untie_emb:",self.args.untie_emb) # 0
            # print("self.args.head_tuning:",self.args.head_tuning) # 0
            # print("self.args.no_auto_device:",self.args.no_auto_device) # 0
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                # Head tuning
                from ht_opt import OPTForCausalLM
                model = OPTForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            elif self.args.no_auto_device or not has_cuda:
                # Honor requested precision in the local CPU integration test too.
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    torch_dtype=torch.float16 if self.args.load_float16 else torch.float32,
                )
            else: # 1
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                if (self.args.parallel is not None )and (self.args.parallel>1):
                        # pdb.set_trace()
                        # from transformers.models.opt.modeling_opt import OPTForCausalLM
                        model = ParallelOPTForCausalLM.from_pretrained(
                        # model = OPTForCausalLM.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        offload_folder="offload",
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{free_in_GB-1}GB' for i in range(torch.cuda.device_count())},
                        load_in_8bit=self.args.load_int8,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        # offload_folder="offload",
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{free_in_GB-1}GB' for i in range(torch.cuda.device_count())},
                        load_in_8bit=self.args.load_int8,
                    )
            model.eval()    # self.model_wrapped

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0
        
        if "llama" in self.args.model_name:
            # LLaMA padding token
            tokenizer.pad_token_id = 0 # technically <unk>

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer


    def forward(self, input_ids, option_len=None, generation=False,**kwargs):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        # n = kwargs.pop("n", 1)
        # perturb = kwargs.pop("perturb", None)
        # self.model._custom_n = n
        # self.model._custom_perturb = perturb
        
        
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids,**kwargs).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification
        if self.args.method in ("MeZO", "ZO-AdaMM", "ZO-RMSProp", "R-AdaZO", "R-AdaZO-Official"):
            from mezotrainer import OurTrainer
            trainer = OurTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
                # compute_metrics = compute_metrics,
            )
        elif self.args.method in ("FZOO", "FZOO-AdaMM", "FZOO-R-AdaZO", "FZOO-Decoupled"):
            if self.args.D == 0 :
                from trainer import OurTrainer
                trainer = OurTrainer(
                    model=self.model, 
                    args=self.args,
                    train_dataset=train_dataset, 
                    eval_dataset=eval_dataset,
                    tokenizer=self.tokenizer,
                    data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
                    # compute_metrics = compute_metrics,
                ) 
            else:
                from trainer import OurTrainer
                trainer = OurTrainer(
                    model=self.model, 
                    args=self.args,
                    train_dataset=train_dataset, 
                    eval_dataset=eval_dataset,
                    tokenizer=self.tokenizer,
                    data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
                ) 
             
        else:
            raise NotImplementedError
        
        
        if self.args.zo_stop_step > 0:
            trainer.add_callback(DiagnosticStopCallback(self.args.zo_stop_step))
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=last_checkpoint) 

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model
        self.trainer = trainer
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag

# def filter_extra_kwargs_hook(module, inputs):  
#     if len(inputs) == 2 and isinstance(inputs[1], dict):
#         if isinstance(module, OptimizedLinear):
#             return inputs
#         kwargs = inputs[1].copy()
#         kwargs.pop('n', None)
#         kwargs.pop('perturb', None)
#         return (inputs[0], kwargs)
#     return inputs

def main():
    args = parse_args()

    set_seed(args.seed)
    task = get_task(args.task_name)
    # print("task:",task)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)
    # for module in framework.modules():### test hook
    #     module.register_forward_pre_hook(filter_extra_kwargs_hook)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples)
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
                    # FZOO-M: also evaluate the best-dev-loss checkpoint, if tracked
                    _tr = framework.trainer
                    if _tr is not None and getattr(_tr, "zo_best_ckpt", False) and _tr._best_state_dict is not None:
                        logger.info("Evaluating best-dev-loss checkpoint (step %d, eval_loss %.6f)" % (_tr._best_step, _tr._best_eval_loss))
                        _final_state = {k: v.detach().to("cpu", copy=True) for k, v in framework.model.named_parameters() if v.requires_grad}
                        _tr.load_best_state()
                        best_metrics = framework.evaluate([], eval_samples)
                        for m in best_metrics:
                            metrics["bestckpt_" + m] = best_metrics[m]
                        if dev_samples is not None:
                            best_dev_metrics = framework.evaluate([], dev_samples)
                            for m in best_dev_metrics:
                                metrics["bestckpt_dev_" + m] = best_dev_metrics[m]
                        metrics["bestckpt_step"] = _tr._best_step
                        metrics["bestckpt_eval_loss"] = _tr._best_eval_loss
                        framework.model.load_state_dict(_final_state, strict=False)
                        del _final_state
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)

#             import numpy as np

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     accuracy = (predictions == labels).mean()
#     print("accuracy:",accuracy)
#     return {"accuracy": accuracy}

if __name__ == "__main__": 
    main()
