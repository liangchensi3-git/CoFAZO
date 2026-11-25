# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import hashlib
from metrics import f1
import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    get_reporting_integration_callbacks,
    hp_params,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

# transformers>=4.48 removed these two; stub them (hp search / fairscale unused here)
def default_hp_search_backend():
    return None

def is_fairscale_available():
    return False

import numpy as np
import torch
import torch.distributed as dist
from adaptive_zo import (
    FZOO_ADAPTIVE_METHODS,
    ZO_DIRECTIONS,
    adaptive_fzoo_decoupled_update,
    adaptive_fzoo_update,
    adaptive_zo_betas,
    fzoo_uses_m2,
    sample_zo_direction,
)
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
# transformers>=4.48 removed transformers.deepspeed; stub (deepspeed unused in this env)
def deepspeed_init(*a, **k):
    raise RuntimeError("deepspeed not available in this environment")
def is_deepspeed_zero3_enabled():
    return False
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
# removed in transformers>=4.48; constants for torch>=2.x
is_torch_greater_or_equal_than_1_10 = True
is_torch_less_than_1_11 = False
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments


# ---- compat stubs for transformers>=4.48 (removed symbols) ----
from enum import Enum
class ShardedDDPOption(str, Enum):
    SIMPLE = "simple"
    ZERO_DP_2 = "zero_dp_2"
    ZERO_DP_3 = "zero_dp_3"
    OFF = "off"

def default_hp_space(self=None):
    return {}
# ---- end compat stubs ----

from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers


_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class ZOEarlyStopCallback(TrainerCallback):
    """FZOO-M: stop training when dev eval_loss rebounds above the running best.

    Triggers when eval_loss > best*(1+thresh) for `patience` consecutive
    evaluations; also stops immediately on non-finite loss (ZO blow-up).
    The final model is then within `thresh` of the best point, which is the
    "cut off the rebound tail" semantics we want for late-stage overfitting.
    """
    def __init__(self, thresh, patience=2):
        self.thresh = thresh
        self.patience = patience
        self.best = None
        self.bad = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        loss = metrics["eval_loss"]
        if not math.isfinite(loss):
            logger.info(f"EARLYSTOP step={state.global_step} non-finite eval_loss={loss}; stopping")
            control.should_training_stop = True
            return
        if self.best is None or loss < self.best:
            self.best = loss
            self.bad = 0
        elif loss > self.best * (1 + self.thresh):
            self.bad += 1
            logger.info(f"EARLYSTOP watch step={state.global_step} eval_loss={loss:.4f} "
                        f"best={self.best:.4f} strikes={self.bad}/{self.patience}")
            if self.bad >= self.patience:
                logger.info(f"EARLYSTOP step={state.global_step} dev loss rebounded "
                            f">{self.thresh:.0%} above best={self.best:.4f}; stopping")
                control.should_training_stop = True
        else:
            self.bad = 0


class OurTrainer(Trainer):
    # transformers>=4.48's Trainer.__init__ no longer sets these legacy attrs
    # that the vendored old-style methods still reference; default them off.
    sharded_ddp = None
    fsdp = None


    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.rand_values = None
        self.zo_random_seeds = None
        self.N = self.args.N
        self.losses = None
        self.thr_step      = self.args.thre
        self.current_step     = 0
        self.current_lr    = self.args.learning_rate
        # FZOO-M: momentum buffer (lazily initialized on first zo_update)
        self.momentum_buffer = None
        self.adaptive_zo_state = None
        self.adaptive_zo_t = 0
        self.zo_momentum     = getattr(self.args, "zo_momentum", 0.0)
        # R13 estimator and optimizer diagnostics. Defaults reproduce R12.
        self.zo_direction = getattr(self.args, "zo_direction", "rademacher")
        self.zo_difference = getattr(self.args, "zo_difference", "single")
        self.zo_sigma_normalize = getattr(self.args, "zo_sigma_normalize", 1) > 0
        self.zo_adam_bias_correction = getattr(self.args, "zo_adam_bias_correction", 0) > 0
        self.zo_update_diag = getattr(self.args, "zo_update_diag", 0) > 0
        if self.zo_direction not in ZO_DIRECTIONS:
            raise ValueError(f"Unknown ZO direction distribution: {self.zo_direction}")
        if self.zo_difference not in ("single", "central"):
            raise ValueError(f"Unknown ZO difference scheme: {self.zo_difference}")
        # FO oracle optimizer states (adam/rmsprop) and ZO-RC row/col v states
        self.fo_m = None
        self.fo_v = None
        self.fo_t = 0
        self.rc_v = None
        self.rc_t = 0
        # FZOO-M: EMA-smoothed sigma normalizer (scalar Adam-v analog)
        self.zo_sigma_ema    = getattr(self.args, "zo_sigma_ema", 0.0)
        self._sigma_ema_v    = 0.0
        self._sigma_ema_t    = 0
        # FZOO-M: early stop on dev-loss rebound
        _es_thresh = getattr(self.args, "zo_early_stop_thresh", 0.0)
        if _es_thresh > 0:
            self.add_callback(ZOEarlyStopCallback(_es_thresh, getattr(self.args, "zo_early_stop_patience", 2)))
        # FZOO-M: in-memory best-dev-loss checkpoint for post-hoc evaluation
        self.zo_best_ckpt = getattr(self.args, "zo_best_ckpt", 0) > 0
        self._best_eval_loss = float("inf")
        self._best_state_dict = None
        self._best_step = -1

    def _get_train_sampler(self):
        # R13 compares different RNG consumers. Isolate data order from the
        # global torch RNG so N/direction/update choices cannot reorder batches.
        if self.zo_update_diag:
            generator = torch.Generator()
            generator.manual_seed(self.args.seed)
            return RandomSampler(self.train_dataset, generator=generator)
        return super()._get_train_sampler()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if metric_key_prefix == "eval" and self.zo_best_ckpt and "eval_loss" in metrics:
            if metrics["eval_loss"] < self._best_eval_loss:
                self._best_eval_loss = metrics["eval_loss"]
                self._best_step = self.state.global_step
                self._best_state_dict = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
                logger.info(f"[BESTCKPT] new best eval_loss={self._best_eval_loss:.6f} at step {self._best_step}")
        return metrics

    def load_best_state(self):
        if self._best_state_dict is None:
            return False
        self.model.load_state_dict(self._best_state_dict)
        return True

        
    def _get_learning_rate(self):
        # ZO updates bypass optimizer.step(), so read the scheduler-driven
        # optimizer value instead of the initial LR cached in current_lr.
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            return optimizer.param_groups[0]["lr"]
        return self.current_lr

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        print("_inner_training_loop-------") # enter
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                        
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4 # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial", random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1: # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        # print("args.deepspeed:",args.deepspeed) # None
        
        if args.deepspeed: # 0
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation: # 1
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()

        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        model = self._wrap_model(self.model_wrapped) 


        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)
        

        # print("args.load_best_model_at_end and self.state.best_model_checkpoint is not None:",args.load_best_model_at_end and self.state.best_model_checkpoint is not None) # 0
        for epoch in range(epochs_trained, num_train_epochs):
            # logger.info("***** enter for epoch in range(epochs_trained, num_train_epochs) *****")
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available(): # 0
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            # epoch_iterator = train_dataloader
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # MeZO added: estimate gradient
                if args.trainer == "zo":
                    # logger.info("***** self.zo_step() *****")
                    tr_loss_step = self.zo_step(model, inputs)
                else:
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer == "zo":
                        self.zo_update(model)
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None: # 0
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    ############## MeZO ##############
    # sigma(+-1) loss-Reuse sumN
    
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1): 
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        # for name, param in self.named_parameters_to_optim:
        for (name, param), rand_value in zip(self.named_parameters_to_optim, self.rand_values):
            z = sample_zo_direction(param, self.zo_direction)
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_step(self, model, inputs):
        args = self.args
        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # FZOO adaptive moments are a separate update path. Do not silently
        # combine them with the older momentum/RC/FO branches.
        if args.method in FZOO_ADAPTIVE_METHODS:
            incompatible = (
                getattr(args, "fo_sgd", 0)
                or self.zo_momentum > 0
                or getattr(args, "zo_rc_v", 0.0) > 0
            )
            if incompatible:
                raise ValueError("FZOO adaptive methods are mutually exclusive with fo_sgd, zo_momentum and zo_rc_v")
            if getattr(args, "zo_diag_interval", 0) > 0:
                raise ValueError("Adaptive FZOO requires zo_diag_interval=0; full-model diagnostics can OOM with fp32 moments")

        if self.zo_update_diag:
            logger.info(
                "ZOBATCH step=%d hash=%s",
                self.state.global_step + 1, self._zo_batch_hash(inputs),
            )

        # FZOO-M: first-order oracle — no perturbations, true gradient by backprop
        if getattr(args, "fo_sgd", 0):
            model.train()
            model.zero_grad(set_to_none=True)
            fo_inputs = self._prepare_inputs(inputs)
            fo_loss = self.compute_loss(model, fo_inputs)
            if isinstance(fo_loss, tuple):
                fo_loss = fo_loss[0]
            fo_loss.backward()
            self.fo_grad = [p.grad.detach().clone() for _, p in self.named_parameters_to_optim]
            model.zero_grad(set_to_none=True)
            self._diag_inputs = inputs
            logger.info(f"loss: {fo_loss.detach().cpu()}")
            logger.info(f"current_lr: {self._get_learning_rate():.3e}")
            return fo_loss.detach().cpu()

        self.rand_values = [torch.rand(1).item() for _ in range(len(self.named_parameters_to_optim))]
        if not self.zo_sigma_normalize and self.zo_sigma_ema > 0:
            raise ValueError("zo_sigma_ema requires zo_sigma_normalize=1")
        if self.state.global_step == 0:
            logger.info(
                "ZO estimator: direction=%s, difference=%s, sigma_normalize=%s, N=%d",
                self.zo_direction, self.zo_difference, self.zo_sigma_normalize, self.N,
            )
        # Sample the random seed for sampling z
        self.zo_random_seeds = []
        loss1s = []
        loss2s = [] if self.zo_difference == "central" else None
        perturbTimes = self.N // 2
        if self.losses is None:
            perturbTimes = self.N
        if self.zo_sigma_normalize and perturbTimes < 2 and self.state.global_step == 0:
            logger.warning(
                "Sigma normalization with N=1 is degenerate: population std is zero; "
                "the 1e-12 clamp produces a very large gradient scale. Use N>=2 for sigma diagnostics."
            )
        with torch.no_grad():
            for i in range(perturbTimes):
                self.zo_random_seed = np.random.randint(1000000000) ####
                self.zo_random_seeds.append(self.zo_random_seed)
                self.zo_perturb_parameters(scaling_factor=1)
                loss1 = self.zo_forward(model, inputs).detach()
                loss1s.append(loss1)
                if self.zo_difference == "central":
                    self.zo_perturb_parameters(scaling_factor=-2)
                    loss2 = self.zo_forward(model, inputs).detach()
                    loss2s.append(loss2)
                    self.zo_perturb_parameters(scaling_factor=1)
                else:
                    self.zo_perturb_parameters(scaling_factor=-1)  ###mabe

            loss1s = torch.tensor(loss1s, dtype=torch.float32)
            if self.zo_difference == "central":
                loss2s = torch.tensor(loss2s, dtype=torch.float32)
                std = torch.std(loss1s - loss2s, unbiased=False)
            else:
                std = torch.std(loss1s, unbiased=False)
            loss_tensor = self.zo_forward(model, inputs).detach().cpu()

        # FZOO-M: optionally replace the instantaneous sigma_t with its
        # bias-corrected EMA (one scalar state, zero memory cost)
        raw_std = float(std)
        if self.zo_sigma_ema > 0:
            b2 = self.zo_sigma_ema
            self._sigma_ema_t += 1
            self._sigma_ema_v = b2 * self._sigma_ema_v + (1 - b2) * raw_std ** 2
            sigma = math.sqrt(self._sigma_ema_v / (1 - b2 ** self._sigma_ema_t))
        else:
            sigma = raw_std
        sigma = max(sigma, 1e-12)

        if self.zo_sigma_normalize:
            normalizer = perturbTimes * sigma
        else:
            normalizer = perturbTimes * args.zo_eps
        loss_difference = loss1s - (loss2s if self.zo_difference == "central" else loss_tensor)
        if self.zo_difference == "central":
            normalizer *= 2
        self.projected_grad = loss_difference / normalizer

        if getattr(args, "zo_trace_diag", 0):
            logger.info(
                "ZOEST step=%d l0=%.9e l_plus=%.9e l_minus=%.9e projected=%.9e",
                self.state.global_step + 1,
                float(loss_tensor), float(loss1s[0]), float(loss2s[0]),
                float(self.projected_grad[0]),
            )

        # FZOO-M diagnostics: stash inputs; the actual measurement runs in
        # zo_update, after the momentum buffer absorbs this step's g_hat
        self._diag_inputs = inputs

        logger.info(f"std: {std}")
        if self.zo_sigma_ema > 0:
            logger.info(f"sigma_ema: {sigma:.6e} (raw std: {raw_std:.6e})")
        logger.info(f"loss: {loss_tensor}")
        logger.info(f"current_lr: {self._get_learning_rate():.3e}")
        return loss_tensor

    @staticmethod
    def _zo_batch_hash(inputs):
        """Small data-order fingerprint; does not alter tensors or RNG state."""
        digest = hashlib.sha256()
        for key in ("input_ids", "labels"):
            value = inputs.get(key) if isinstance(inputs, Mapping) else None
            if torch.is_tensor(value):
                digest.update(key.encode("utf-8"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()[:16]

    @staticmethod
    def _stream_cos_norm(a_list, b_list):
        """流式 cos/norm：逐张量累加点积，不做全参数 cat。
        r10 教训：2.7B 模型上 cat 瞬时多吃 ~10GB，45GB 卡直接 OOM。
        返回 (cos, |a|, |b|)；Python float 累加，精度优于 fp32 大向量点积。"""
        dot = 0.0
        na = 0.0
        nb = 0.0
        for a, b in zip(a_list, b_list):
            af = a.detach().float().ravel()
            bf = b.detach().float().ravel()
            dot += float(torch.dot(af, bf))
            na += float(torch.dot(af, af))
            nb += float(torch.dot(bf, bf))
        cos = dot / max(na * nb, 1e-300) ** 0.5
        return cos, na ** 0.5, nb ** 0.5

    def _fo_diag(self):
        """FO oracle diagnostics: true-grad norm, and (with momentum on) the
        cosine / norm of the EMA momentum buffer vs the current true gradient.
        This is the first-order reference curve for the ZO DIAG cos_m numbers
        (ZO: ~0.0008); the gap between the two is the momentum information
        loss of the ZO estimate. Lines prefixed FODIAG for grepping.
        """
        if self.momentum_buffer is not None:
            cos, nm, ng = self._stream_cos_norm(self.momentum_buffer, self.fo_grad)
            cos = max(-1.0, min(1.0, cos))
            logger.info(f"FODIAG step={self.state.global_step} norm_g={ng:.3e} cos_m={cos:.4f} norm_m={nm:.3e}")
        else:
            _, _, ng = self._stream_cos_norm(self.fo_grad, self.fo_grad)
            logger.info(f"FODIAG step={self.state.global_step} norm_g={ng:.3e}")

    def _zo_rc_diag(self, model, inputs, g_hat, dirs):
        """FZOO-RC diagnostics: cos(g_hat, g_true) vs cos(preconditioned dir,
        g_true) — 直接回答"行/列 v 预条件是否改善了更新方向"。RCDIAG 前缀。
        调用时参数尚未更新(与 _zo_diag 一致的口径)。"""
        model.eval()
        model.zero_grad(set_to_none=True)
        di = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, di)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        g_true = [p.grad.detach().float() for _, p in self.named_parameters_to_optim]
        model.zero_grad(set_to_none=True)

        def flat_cos(a_list, b_list):
            c, _, _ = OurTrainer._stream_cos_norm(a_list, b_list)
            if math.isnan(c):  # 发散后如实上报; min/max 遇 NaN 会谎报成 1.0 (r7b 教训)
                return float("nan")
            return max(-1.0, min(1.0, c))

        logger.info(f"RCDIAG step={self.state.global_step} "
                    f"cos_ghat={flat_cos(g_hat, g_true):.6f} cos_pre={flat_cos(dirs, g_true):.6f}")
        del g_true

    def _zo_diag(self, model, inputs):
        """Diagnostic (for the momentum ablation): quality of the ZO update direction.

        Computes the true gradient on the current batch with one backprop pass
        (params are still unperturbed when this is called from zo_update), then
        logs cosine similarity and norms for the ZO estimate g_hat and, when
        momentum is active, the momentum buffer m about to be applied. Lines
        are prefixed DIAG for grepping. Intended for small models: holds full
        fp32 grads + g_hat in memory.
        """
        def flat_cos(a_list, b_list):
            return self._stream_cos_norm(a_list, b_list)

        # 1) true gradient on the same batch
        model.eval()
        model.zero_grad(set_to_none=True)
        diag_inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, diag_inputs)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        g_true = [p.grad.detach().float().clone() for _, p in self.named_parameters_to_optim]
        model.zero_grad(set_to_none=True)

        # 2) rebuild the ZO estimate g_hat by replaying the perturbation seeds
        g_hat = [torch.zeros(p.data.size(), dtype=torch.float32, device=p.data.device)
                 for _, p in self.named_parameters_to_optim]
        for idx, zo_random_seed in enumerate(self.zo_random_seeds):
            torch.manual_seed(zo_random_seed)
            alpha = float(self.projected_grad[idx])
            for gh, (_, param) in zip(g_hat, self.named_parameters_to_optim):
                z = sample_zo_direction(param, self.zo_direction)
                gh.add_(z.float(), alpha=alpha)

        cos_g, norm_g, norm_true = flat_cos(g_hat, g_true)
        msg = (f"DIAG step={self.state.global_step} cos_ghat={cos_g:.4f} "
               f"norm_ghat={norm_g:.3e} norm_gtrue={norm_true:.3e}")
        if self.momentum_buffer is not None:
            cos_m, norm_m, _ = flat_cos([b.float() for b in self.momentum_buffer], g_true)
            msg += f" cos_m={cos_m:.4f} norm_m={norm_m:.3e}"
        logger.info(msg)
        self._zo_struct_diag(g_true)
        del g_true, g_hat

    def _zo_struct_diag(self, g_true):
        """Structural statistics of the true gradient (feasibility probes for a
        ZO second moment v): (1) layerwise energy shares vs the ZO noise floor,
        (2) row/column energy concentration (Gini) of 2D matrices.
        Lines prefixed DIAG2 for grepping. All stats are computed from g_true
        only, so no extra forward/backward cost.
        """
        import math
        named = list(self.named_parameters_to_optim)
        total_sq = float(sum(float(g.pow(2).sum()) for g in g_true))
        if total_sq <= 0:
            return
        n_params = int(sum(g.numel() for g in g_true))
        N = max(int(getattr(self.args, "N", 8)), 1)

        def group_of(name):
            # OPT-style names: model.decoder.layers.7.self_attn.q_proj.weight
            if ".layers." in name:
                return "L" + name.split(".layers.")[1].split(".")[0]
            for key in ("embed_tokens", "embed_positions", "layer_norm", "lm_head"):
                if key in name:
                    return key
            return ".".join(name.split(".")[:3])

        # (1) layerwise energy share, per-parameter density, and debiased share.
        # ZO noise floor for group l: E||g_hat_l||^2 noise part ~ d_l*||g||^2/N.
        groups = {}
        for (name, _), g in zip(named, g_true):
            key = group_of(name)
            e = float(g.pow(2).sum())
            if key not in groups:
                groups[key] = [0.0, 0]
            groups[key][0] += e
            groups[key][1] += g.numel()
        parts = []
        for key in sorted(groups, key=lambda k: -groups[k][0]):
            e, d_l = groups[key]
            share = e / total_sq
            density = share / (d_l / n_params)  # 1.0 = uniform per-param energy
            floor = (d_l * total_sq / N) / total_sq  # noise floor as share of total
            debiased = share - floor  # rough signal share above noise floor
            parts.append(f"{key}:{share:.4f}/{density:.2f}/{debiased:+.4f}")
        logger.info(f"DIAG2 step={self.state.global_step} layer share/density/debiased(N={N}): " + " ".join(parts))

        # (2) row/col Gini + top-1% energy share for 2D matrices
        def gini(x):
            xs = torch.sort(x.flatten())[0]
            n = xs.numel()
            if float(xs.sum()) <= 0:
                return 0.0
            idx = torch.arange(1, n + 1, dtype=xs.dtype, device=xs.device)
            return float((2.0 * (idx * xs).sum() / (n * xs.sum())) - (n + 1.0) / n)

        row_g, col_g = [], []
        for (name, _), g in zip(named, g_true):
            if g.dim() != 2 or min(g.shape) < 8:
                continue
            re = g.pow(2).sum(dim=1)
            ce = g.pow(2).sum(dim=0)
            row_g.append(gini(re))
            col_g.append(gini(ce))
        if row_g:
            logger.info(
                f"DIAG2 step={self.state.global_step} rowGini mean={sum(row_g)/len(row_g):.3f} "
                f"max={max(row_g):.3f} | colGini mean={sum(col_g)/len(col_g):.3f} max={max(col_g):.3f} "
                f"({len(row_g)} matrices)")


    def zo_update(self, model):
        args = self.args
        lr = self._get_learning_rate()
        diag_due = (getattr(args, "zo_diag_interval", 0) > 0
                    and self.state.global_step % args.zo_diag_interval == 0)

        # FZOO-M: first-order oracle update (SGD / SGD + EMA momentum / Adam / RMSProp)
        if getattr(args, "fo_sgd", 0):
            opt = getattr(args, "fo_optimizer", "sgd")
            if opt in ("adam", "rmsprop"):
                # r7a: 分离 Adam 收益中 m 与 v 的贡献。adam: β1=0.9; rmsprop: β1=0。
                b1 = 0.9 if opt == "adam" else 0.0
                b2, eps = 0.999, 1e-8
                if self.fo_m is None:
                    self.fo_m = [torch.zeros(p.data.size(), dtype=torch.float32, device=p.data.device)
                                 for _, p in self.named_parameters_to_optim]
                    self.fo_v = [torch.zeros(p.data.size(), dtype=torch.float32, device=p.data.device)
                                 for _, p in self.named_parameters_to_optim]
                    self.fo_t = 0
                    logger.info(f"FO oracle optimizer={opt}: b1={b1}, b2={b2}, eps={eps}")
                self.fo_t += 1
                bc1 = 1 - b1 ** self.fo_t
                bc2 = 1 - b2 ** self.fo_t
                for m_, v_, g in zip(self.fo_m, self.fo_v, self.fo_grad):
                    gf = g.float()
                    m_.mul_(b1).add_(gf, alpha=1 - b1)
                    v_.mul_(b2).addcmul_(gf, gf, value=1 - b2)
                if diag_due:
                    self._fo_diag()
                for (name, param), m_, v_ in zip(self.named_parameters_to_optim, self.fo_m, self.fo_v):
                    d = (m_ / bc1) / ((v_ / bc2).sqrt() + eps)
                    param.data.add_(d.to(param.data.dtype), alpha=-lr)
            else:
                beta = self.zo_momentum
                if beta > 0:
                    if self.momentum_buffer is None:
                        self.momentum_buffer = [torch.zeros(p.data.size(), dtype=torch.float32, device=p.data.device)
                                                for _, p in self.named_parameters_to_optim]
                        logger.info(f"FO oracle momentum enabled: beta={beta}, buffered params={len(self.momentum_buffer)}")
                    # r10b: momentum_style=heavyball 用经典形式 buf=β·buf+g(有效步长放大
                    # 1/(1-β)); 默认 ema 保持 r5/r6 历史行为(见 momentum_code_audit_20261024)。
                    gscale = 1 - beta if getattr(args, "momentum_style", "ema") == "ema" else 1.0
                    for buf, g in zip(self.momentum_buffer, self.fo_grad):
                        buf.mul_(beta).add_(g.float(), alpha=gscale)
                    if diag_due:
                        self._fo_diag()
                    for (name, param), buf in zip(self.named_parameters_to_optim, self.momentum_buffer):
                        param.data.add_(buf.to(param.data.dtype), alpha=-lr)
                else:
                    if diag_due:
                        self._fo_diag()
                    for (name, param), g in zip(self.named_parameters_to_optim, self.fo_grad):
                        param.data.add_(g, alpha=-lr)
            self.lr_scheduler.step()
            return

        # FZOO + ZO-AdaMM/R-AdaZO: replay the N Rademacher directions into
        # fp32 m/v states. The two methods differ only in v=g^2 versus v=m^2.
        if args.method in FZOO_ADAPTIVE_METHODS:
            beta1, beta2 = adaptive_zo_betas(args.method, args.zo_beta1, args.zo_beta2)
            first_step = self.adaptive_zo_state is None
            if first_step:
                logger.info(
                    "Adaptive FZOO enabled: method=%s, beta1=%s, beta2=%s, eps=%s, N=%d, "
                    "trainable_tensors=%d, bias_correction=%s, g2_warmup=%d, "
                    "g2_after=%d, decoupled_direction=%s, decoupled_norm=%s",
                    args.method, beta1, beta2, args.zo_adam_eps, len(self.zo_random_seeds),
                    len(self.named_parameters_to_optim), self.zo_adam_bias_correction,
                    getattr(args, "zo_radazo_g2_warmup", 0),
                    getattr(args, "zo_radazo_g2_after", 0),
                    getattr(args, "zo_decoupled_direction", "m2"),
                    getattr(args, "zo_decoupled_norm", "m2"),
                )
            self.adaptive_zo_t += 1
            update_diag = (
                {} if self.zo_update_diag and (
                    self.adaptive_zo_t <= 20 or self.adaptive_zo_t % 10 == 0
                ) else None
            )
            update_kwargs = dict(
                named_parameters=self.named_parameters_to_optim,
                zo_random_seeds=self.zo_random_seeds,
                projected_grad=self.projected_grad,
                learning_rate=lr,
                beta1=beta1,
                beta2=beta2,
                eps=args.zo_adam_eps,
                state=self.adaptive_zo_state,
                direction=self.zo_direction,
                bias_correction=self.zo_adam_bias_correction,
                step=self.adaptive_zo_t,
                diagnostics=update_diag,
            )
            if args.method == "FZOO-Decoupled":
                self.adaptive_zo_state = adaptive_fzoo_decoupled_update(
                    direction_kind=args.zo_decoupled_direction,
                    norm_kind=args.zo_decoupled_norm,
                    **update_kwargs,
                )
            else:
                self.adaptive_zo_state = adaptive_fzoo_update(
                    refined_second_moment=fzoo_uses_m2(
                        args.method,
                        self.adaptive_zo_t,
                        getattr(args, "zo_radazo_g2_warmup", 0),
                        getattr(args, "zo_radazo_g2_after", 0),
                    ),
                    **update_kwargs,
                )
            if update_diag is not None and (
                self.adaptive_zo_t <= 20 or self.adaptive_zo_t % 10 == 0
            ):
                count = max(update_diag["numel"], 1)
                logger.info(
                    "ZODIAG step=%d g_rms=%.6e g_l2=%.6e m_rms=%.6e sqrtv_rms=%.6e "
                    "update_rms=%.6e update_l2=%.6e update_max=%.6e "
                    "actual_delta_rms=%.6e actual_delta_l2=%.6e actual_delta_max=%.6e",
                    self.adaptive_zo_t,
                    (update_diag["g_sq"] / count) ** 0.5, update_diag["g_sq"] ** 0.5,
                    (update_diag["m_sq"] / count) ** 0.5,
                    (update_diag["sqrt_v_sq"] / count) ** 0.5,
                    (update_diag["update_sq"] / count) ** 0.5, update_diag["update_sq"] ** 0.5,
                    update_diag["update_max_abs"],
                    (update_diag["actual_delta_sq"] / count) ** 0.5,
                    update_diag["actual_delta_sq"] ** 0.5,
                    update_diag["actual_delta_max_abs"],
                )
                cos_mprev_g = 0.0
                cos_mprev_m = 0.0
                if update_diag["m_prev_sq"] > 0 and update_diag["g_sq"] > 0:
                    cos_mprev_g = update_diag["m_prev_g_dot"] / (
                        update_diag["m_prev_sq"] * update_diag["g_sq"]
                    ) ** 0.5
                if update_diag["m_prev_sq"] > 0 and update_diag["m_raw_sq"] > 0:
                    cos_mprev_m = update_diag["m_prev_m_dot"] / (
                        update_diag["m_prev_sq"] * update_diag["m_raw_sq"]
                    ) ** 0.5
                logger.info(
                    "ZOMOM step=%d cos_mprev_g=%.6e cos_mprev_m=%.6e sign_mprev_g=%.6e",
                    self.adaptive_zo_t,
                    cos_mprev_g,
                    cos_mprev_m,
                    update_diag["sign_agree"] / max(update_diag["sign_valid"], 1),
                )
                logger.info(
                    "ZOPARAM step=%d probe_numel=%d expected_l2=%.6e observed_l2=%.6e "
                    "roundoff_max=%.6e expected_nonzero_observed_zero=%d",
                    self.adaptive_zo_t,
                    update_diag["probe_numel"],
                    update_diag["probe_expected_delta_sq"] ** 0.5,
                    update_diag["probe_observed_delta_sq"] ** 0.5,
                    update_diag["probe_roundoff_max_abs"],
                    update_diag["probe_expected_nonzero_observed_zero"],
                )
                if args.method == "FZOO-Decoupled":
                    direction_sq = update_diag[f"direction_{args.zo_decoupled_direction}_sq"]
                    target_sq = update_diag[f"direction_{args.zo_decoupled_norm}_sq"]
                    direction_cos = 0.0
                    if update_diag["direction_m2_sq"] > 0 and update_diag["direction_g2_sq"] > 0:
                        direction_cos = update_diag["direction_dot"] / (
                            update_diag["direction_m2_sq"] * update_diag["direction_g2_sq"]
                        ) ** 0.5
                    logger.info(
                        "ZONORM step=%d direction=%s norm=%s scale=%.6e "
                        "dir_l2=%.6e target_l2=%.6e cos_m2_g2=%.6e",
                        self.adaptive_zo_t,
                        args.zo_decoupled_direction,
                        args.zo_decoupled_norm,
                        update_diag["norm_scale"],
                        direction_sq ** 0.5,
                        target_sq ** 0.5,
                        direction_cos,
                    )
            self.lr_scheduler.step()
            return

        # FZOO-RC: 行/列分解二阶矩预条件(Adafactor 式 v, r7b 主方法)
        rc_beta2 = getattr(args, "zo_rc_v", 0.0)
        rc_warmup = getattr(args, "zo_rc_warmup", 50)
        rc_vfloor = getattr(args, "zo_rc_vfloor", 0.01)
        if rc_beta2 > 0:
            # 1) 回放扰动种子, 显式拼出 g_hat(fp32 临时缓冲, 每步用后释放)
            g_hat = [torch.zeros(p.data.size(), dtype=torch.float32, device=p.data.device)
                     for _, p in self.named_parameters_to_optim]
            for idx, zo_random_seed in enumerate(self.zo_random_seeds):
                torch.manual_seed(zo_random_seed)
                alpha = float(self.projected_grad[idx])
                for gh, ((name, param), rand_value) in zip(g_hat, zip(self.named_parameters_to_optim, self.rand_values)):
                    z = sample_zo_direction(param, self.zo_direction)
                    gh.add_(z.float(), alpha=alpha)
            # 2) 去偏行/列能量: E[g_hat_ij^2] = g_ij^2 + sum(alpha_i^2), 减去已知噪声项。
            #    r9 修正: 去偏观测不再逐步 clamp —— clamp(min=0) 砍掉涨落负半会引入
            #    系统性正偏(≈0.4*噪声涨落, 与 1.8% 信号同量级)且不随 EMA 消失,
            #    r8 阴性即此偏置洗掉 vhat 对比度所致。改为 EMA 累积原始(可负)观测,
            #    构造 vhat 时才 clamp, 恢复无偏性, 让 EMA 的平均降噪红利生效。
            noise = float(sum(float(a) ** 2 for a in self.projected_grad))
            if self.rc_v is None:
                self.rc_v = [None] * len(self.named_parameters_to_optim)
                logger.info(f"FZOO-RC row/col v enabled: beta2={rc_beta2}")
            self.rc_t += 1
            bc = 1 - rc_beta2 ** self.rc_t
            for k, ((name, param), gh) in enumerate(zip(self.named_parameters_to_optim, g_hat)):
                if gh.dim() == 2 and min(gh.shape) >= 8:
                    r_obs = gh.pow(2).mean(dim=1) - noise
                    c_obs = gh.pow(2).mean(dim=0) - noise
                    if self.rc_v[k] is None:
                        self.rc_v[k] = [torch.zeros_like(r_obs), torch.zeros_like(c_obs)]
                    self.rc_v[k][0].mul_(rc_beta2).add_(r_obs, alpha=1 - rc_beta2)
                    self.rc_v[k][1].mul_(rc_beta2).add_(c_obs, alpha=1 - rc_beta2)
            # 3) 预条件方向: dir = g_hat / sqrt(v_hat), v_hat 归一到均值 1。
            #    关键: ZO 噪声各向同性而 v 高度集中, 直接除会在低 v 坐标上把噪声
            #    放大约 100 倍(本地冒烟 loss 爆炸实测)。因此整体 RMS 匹配回 g_hat:
            #    步长统计与 base 完全一致, v 只改变方向(逐坐标相对权重)。
            #    r7b 事故教训: 去偏 clamp 使过半 vhat 条目为精确 0, sqrt(0)+1e-8
            #    产生 1e8 倍放大, fp16 cast 溢出 → 首步即 NaN。修复: 均值归一后
            #    加地板 vfloor(放大硬限 1/sqrt(vfloor)=10x, 预条件信任域);
            #    warmup 期内 v 未积累, 回退 base ZO 方向。
            in_warmup = self.rc_t <= rc_warmup
            rc_oneside = getattr(args, "zo_rc_oneside", 0) > 0
            dirs = []
            ss_g, ss_d = 0.0, 0.0
            vhat_stats = []  # (max/median, 贴地板比例) per 2D matrix, r9 对比度仪器
            for k, ((name, param), gh) in enumerate(zip(self.named_parameters_to_optim, g_hat)):
                if self.rc_v[k] is not None and not in_warmup:
                    vr = (self.rc_v[k][0] / bc).clamp_(min=0.0)  # r9: clamp 移到 EMA 之后
                    vc = (self.rc_v[k][1] / bc).clamp_(min=0.0)
                    vhat = torch.outer(vr, vc) / vr.mean().clamp(min=1e-20)
                    vhat = vhat / vhat.mean().clamp(min=1e-20)
                    # r10 v3 只缩不放: 高 v 坐标降权, 低 v 坐标保持原样。
                    # r9 教训: v2 地板 0.01 把 75% 无信号坐标放大 10x, FO 下放大死坐标
                    # 无害(g=0)但 ZO 下放大的是纯噪声, cos_pre 稳定低于 cos_ghat。
                    lo = 1.0 if rc_oneside else rc_vfloor
                    vhat = vhat.clamp_(min=lo)
                    if diag_due:
                        vhat_stats.append((
                            float(vhat.max() / vhat.median().clamp(min=1e-20)),
                            float((vhat <= lo * 1.0001).float().mean()),
                        ))
                    d = gh / (vhat.sqrt() + 1e-8)
                else:
                    d = gh
                dirs.append(d)
                ss_g += float(gh.pow(2).sum())
                ss_d += float(d.pow(2).sum())
            rms_match = math.sqrt(ss_g / max(ss_d, 1e-30))
            if diag_due:
                if vhat_stats:
                    cm = sum(s[0] for s in vhat_stats) / len(vhat_stats)
                    ff = sum(s[1] for s in vhat_stats) / len(vhat_stats)
                    logger.info(f"RCV step={self.state.global_step} "
                                f"contrast_max_med={cm:.2f} frac_floor={ff:.3f} nmat={len(vhat_stats)}")
                self._zo_rc_diag(model, self._diag_inputs, g_hat, dirs)
            for (name, param), d in zip(self.named_parameters_to_optim, dirs):
                param.data.add_((d * (-lr * rms_match)).to(param.data.dtype))
            del g_hat, dirs
            self.lr_scheduler.step()
            return

        if self.zo_momentum > 0:
            # FZOO-M: momentum on the normalized direction (g_hat / sigma_t).
            # m = beta*m + (1-beta)*g_hat/sigma_t ; theta -= lr * m
            beta = self.zo_momentum
            gscale = 1 - beta if getattr(args, "momentum_style", "ema") == "ema" else 1.0
            if self.momentum_buffer is None:
                self.momentum_buffer = [torch.zeros_like(param.data) for _, param in self.named_parameters_to_optim]
                logger.info(f"FZOO-M momentum enabled: beta={beta}, buffered params={len(self.momentum_buffer)}")
            for buf in self.momentum_buffer:
                buf.mul_(beta)
            for idx, zo_random_seed in enumerate(self.zo_random_seeds):
                # Reset the random seed to replay the same z as the perturbation
                torch.manual_seed(zo_random_seed)
                alpha = gscale * float(self.projected_grad[idx])
                for buf, ((name, param), rand_value) in zip(self.momentum_buffer, zip(self.named_parameters_to_optim, self.rand_values)):
                    z = sample_zo_direction(param, self.zo_direction)
                    buf.add_(z, alpha=alpha)
            if diag_due:
                self._zo_diag(model, self._diag_inputs)
            for (name, param), buf in zip(self.named_parameters_to_optim, self.momentum_buffer):
                param.data = param.data - lr * buf
        else:
            if diag_due:
                self._zo_diag(model, self._diag_inputs)
            for idx, zo_random_seed in enumerate(self.zo_random_seeds):
                # Reset the random seed for sampling zs

                # if idx not in self.topk_indices:
                #   continue
                torch.manual_seed(zo_random_seed)
                for (name, param), rand_value in zip(self.named_parameters_to_optim, self.rand_values):
                    # z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    z = sample_zo_direction(param, self.zo_direction)
                    param.data = param.data - lr * (self.projected_grad[idx] * z)

        self.lr_scheduler.step()

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        # print("self.args.non_diff:",self.args.non_diff) # False
        # print("self.args.n_gpu:",self.args.n_gpu) # 1
        # exit()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) # outputs = model(**inputs)
            if self.args.n_gpu > 1: # 0
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        # exit()
        return loss.detach()


    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)


    # original:

    ############## Misc overload functions ##############


    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
