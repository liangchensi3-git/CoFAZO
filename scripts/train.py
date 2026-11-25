"""Run one paper configuration on a local Qwen3 Base checkpoint."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# Training uses 1,000 examples and 100 held-out development examples in every task.
TASKS = {
    "sst2": ("SST2", 872), "trec": ("TREC", 500), "cr": ("CR", 376),
    "cola": ("CoLA", 1043), "rte": ("RTE", 277), "mnli": ("MNLI", 19647),
    "subj": ("SUBJ", 2000),
}
# Values are the exact arm, second-moment mode and codec names of the recorded runs.
METHODS = {
    "fzoo": ("fzoo", "fzoo", "none"),
    "dense": ("stored_m", "full", "none"),
    "factored": ("stored_m", "rc", "none"),
    "cofazo": ("all_rc4", "rc", "signed_rc4"),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Path to a local Qwen3 Base checkpoint")
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "prepared")
    p.add_argument("--dataset", choices=TASKS, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, help="Stop after this many updates on the original 10,000-step schedule")
    p.add_argument("--eval-steps", type=int, help="Evaluation interval, normally 250 updates")
    p.add_argument("--output", type=Path, required=True, help="New directory for results and logs")
    p.add_argument("--gpu", type=int, help="Select one GPU from the visible devices (default: first visible)")
    p.add_argument("--cpu-test", action="store_true", help="Small local model check using 8/4/4 examples")
    p.add_argument("--dry-run", action="store_true", help="Print effective configuration and command")
    return p


def configuration(args):
    task, neval = TASKS[args.dataset]
    arm, kind, codec = METHODS[args.method]
    steps = args.steps if args.steps is not None else (8 if args.cpu_test else 10000)
    interval = args.eval_steps if args.eval_steps is not None else (2 if args.cpu_test else 250)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if not 1 <= steps <= 10000 or interval < 1 or steps % interval:
        raise ValueError("Choose 1–10,000 steps divisible by the evaluation interval")
    if args.cpu_test and steps > 8:
        raise ValueError("Use at most eight steps for a CPU fixture")
    return dict(dataset=args.dataset, task=task, method=args.method, arm=arm, kind=kind, codec=codec,
                learning_rate=args.learning_rate, seed=args.seed, steps=steps, eval_steps=interval,
                ntrain=8 if args.cpu_test else 1000, ndev=4 if args.cpu_test else 100,
                neval=4 if args.cpu_test else neval, batch=1 if args.cpu_test else 16,
                max_length=64 if args.cpu_test else 2048, logging_steps=1 if args.cpu_test else 50,
                cpu_test=args.cpu_test)


def engine_args(c, model, out):
    args = [
        "--r22-kind", c["kind"], "--m-horizon", 0, "--r22-artifacts", out,
        "--method", "R-AdaZO-Official", "--q", 8, "--load_float16", "--r22-diag-interval", 100,
        "--r22-chunk", 4194304, "--model_name", model, "--task_name", c["task"],
        "--output_dir", out / "checkpoint_unused", "--result_file", out / "result.json",
        "--tag", f"{c['dataset']}_{c['method']}_s{c['seed']}", "--train_set_seed", 0,
        "--seed", c["seed"], "--num_train", c["ntrain"], "--num_dev", c["ndev"], "--num_eval", c["neval"],
        "--max_length", c["max_length"], "--max_steps", 10000, "--zo_stop_step", c["steps"],
        "--logging_steps", c["logging_steps"], "--eval_steps", c["eval_steps"], "--trainer", "zo",
        "--learning_rate", c["learning_rate"], "--zo_eps", 0.0005, "--per_device_train_batch_size", c["batch"],
        "--gradient_accumulation_steps", 1, "--lr_scheduler_type", "linear",
        "--evaluation_strategy", "steps", "--save_strategy", "no", "--train_as_classification",
        "--zo_beta1", 0.9, "--zo_beta2", 0.99, "--zo_adam_eps", 1e-8, "--zo_update_diag", 1,
        "--zo_diag_interval", 0, "--zo_trace_diag", 0, "--zo_best_ckpt", 1, "--report_to", "none",
        "--dataloader_num_workers", 0, "--r28-drop-after", 0, "--r28-drop-factor", 0.25,
        "--r32-no-history", "none", "--r34-m-compression", c["codec"], "--r34-rank", 16, "--r39-arm", c["arm"],
    ]
    return list(map(str, args))


def audit(out, config):
    rows = [json.loads(line) for line in (out / "trace.jsonl").read_text("utf8").splitlines()]
    steps = [r for r in rows if r["kind"] == "step"]
    if [r["step"] for r in steps] != list(range(1, config["steps"] + 1)) or rows[-1]["kind"] != "complete":
        raise ValueError("Incomplete training trace")
    for row in steps:
        expected_lr = config["learning_rate"] * (10001 - row["step"]) / 10000
        if not math.isclose(row["lr"], expected_lr, rel_tol=1e-12):
            raise ValueError("Learning-rate schedule changed")
        if not row["finite"] or row["forward_calls"] != 9 or not math.isfinite(row["loss"]):
            raise ValueError("Invalid update evidence")
    evals = [r for r in rows if r["kind"] == "eval"]
    if [r["step"] for r in evals] != list(range(config["eval_steps"], config["steps"] + 1, config["eval_steps"])):
        raise ValueError("Incomplete development curve")
    result = json.loads((out / "result.json").read_text("utf8"))
    best = min(evals, key=lambda row: row["loss"])
    if result["bestckpt_step"] != best["step"] or not math.isclose(result["bestckpt_eval_loss"], best["loss"], rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Inconsistent development checkpoint")
    for key in ("accuracy", "bestckpt_accuracy", "dev_accuracy", "bestckpt_dev_accuracy"):
        if not 0 <= result[key] <= 1:
            raise ValueError("Invalid accuracy: " + key)
    if config["dataset"] == "mnli":
        records = [json.loads(s) for s in (out / "evaluations41.jsonl").read_text("utf8").splitlines()]
        if len(records) != 4:
            raise ValueError("Expected four final and selected-checkpoint evaluation records")
        for index, prefix in ((0, ""), (2, "bestckpt_")):
            record = records[index]
            if record["count"] != config["neval"] or result[prefix + "accuracy"] != record["metrics"]["accuracy"]:
                raise ValueError("Incorrect pooled MNLI evaluation")
            if not config["cpu_test"] and Counter(record["splits"]) != {"matched": 9815, "mismatched": 9832}:
                raise ValueError("Both complete MNLI evaluation splits are required")
            for split in set(record["splits"]):
                key = split + "_accuracy"
                if result[prefix + key] != record["metrics"][key]:
                    raise ValueError("Inconsistent MNLI split accuracy")
    if not (out / "state_fingerprint.json").is_file():
        raise ValueError("Missing final state fingerprint")
    return result


def select_device(index):
    # Respect CUDA_VISIBLE_DEVICES set by Slurm or a container, including GPU UUIDs.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if index < 0:
        raise ValueError("GPU index must be nonnegative")
    if visible is None:
        return str(index)
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    if index >= len(devices):
        raise ValueError("Requested GPU is outside CUDA_VISIBLE_DEVICES")
    return devices[index]


def main():
    args = parser().parse_args()
    config = configuration(args)
    out, model, data = args.output.resolve(), args.model.resolve(), args.data_root.resolve()
    command = [sys.executable, str(ROOT / "scripts" / "_entry.py"), "--data-root", str(data), *engine_args(config, model, out)]
    if args.dry_run:
        print(json.dumps(dict(configuration=config, command=command), indent=2))
        return
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10 for this research runtime")
    if json.loads((model / "config.json").read_text("utf8")).get("model_type") != "qwen3":
        raise ValueError("Expected a local Qwen3 Base checkpoint")
    if not (data / args.dataset / "dataset_dict.json").is_file():
        raise FileNotFoundError(f"Prepare {args.dataset} under {data} with data/prepare.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu_test else select_device(args.gpu or 0)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                      MKL_NUM_THREADS="4", PYTHONUTF8="1")
    import torch
    import transformers
    if transformers.__version__ != "4.51.3":
        raise RuntimeError("Install the pinned version of Transformers (4.51.3)")
    if not args.cpu_test and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-model training")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(configuration=config, command=command, torch=torch.__version__, transformers=transformers.__version__)
    (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(f"Running {args.dataset} ({args.method}); see {out / 'train.log'}", flush=True)
    try:
        with (out / "train.log").open("w", encoding="utf8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = audit(out, config)
    except Exception as err:
        (out / "status.json").write_text(json.dumps(dict(status="failed", error=str(err))), encoding="utf8")
        raise
    status = dict(status="cpu_test_success" if args.cpu_test else "success", result=result)
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
