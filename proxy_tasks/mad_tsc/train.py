"""Single baseline MAD selective-copying configuration; not the 396-run sweep.

Uses the pinned SDPA MAD train.train implementation, including its repeatedly
monitored 'test' split. That split is development data, not a final test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from common.paths import THIRD_PARTY, task_data, task_output
from .models import MODELS

from .data import DATASET, read_progress


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS.names(), default="mh_attention_swiglu")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--mad_root", type=Path, default=THIRD_PARTY / "mad-lab")
    p.add_argument("--data_dir", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--smoke_test", action="store_true", help="One epoch on the same fixed baseline data")
    return p.parse_args()


def main():
    args = parse_args()
    from common.gpu_guard import require_cuda
    require_cuda("[mad_tsc]")
    import numpy as np
    import random
    import torch

    mad_root = args.mad_root.expanduser().resolve()
    data = (args.data_dir or task_data("mad_tsc")).resolve() / DATASET
    for split in ["train", "test"]:
        for filename in ["inputs.npy", "targets.npy"]:
            if not (data / split / filename).is_file():
                raise FileNotFoundError(data / split / filename)
    out = args.out_dir / f"{args.model}_seed{args.seed}" if args.out_dir else task_output("mad_tsc", args.model, args.seed)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "mad").exists():
        raise FileExistsError(f"Refusing to reuse existing MAD logs: {out / 'mad'}")
    sys.path.insert(0, str(mad_root))
    # Upstream config paths are relative to its checkout.
    os.environ["TUNE_ORIG_WORKING_DIR"] = str(mad_root)
    from mad.configs import MADConfig, MADModelConfig

    spec = importlib.util.spec_from_file_location("mad_upstream_train", mad_root / "train.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    config = MADConfig(task="selective-copying", vocab_size=16, seq_len=256,
                       num_tokens_to_copy=16, num_train_examples=12800, num_test_examples=1280,
                       seed=args.seed, epochs=1 if args.smoke_test else args.epochs,
                       precision="32", accelerator="cuda", devices=1, data_path=str(out / "data"),
                       num_data_workers=0, persistent_data_workers=False)
    # MAD includes model seed in its data path. Alias every seed to the fixed data-seed cache.
    staged = Path(config.dataset_path)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.symlink_to(data, target_is_directory=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = MADModelConfig(layers=MODELS.build(args.model), dim=128, vocab_size=16,
                           max_length=256).build_model_from_registry()
    count = sum(p.numel() for p in model.parameters())
    print(f"[mad_tsc] model={args.model} params={count:,}", flush=True)
    result = upstream.train(model, config, str(out / "mad"), log_to_csv=True,
                            log_to_wandb=False, save_checkpoints=True)
    completed, peak = read_progress(out / "mad/logs/metrics.csv")
    stop = "max_epochs" if completed == config.epochs else "accuracy_threshold"
    raw = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "n_params": count, "history": [{"epoch": completed}],
           "val_accuracy": float(result.iloc[0]["test_acc"]), "stop_reason": stop,
           "development_peak_accuracy": peak,
           "data_seed": 12345, "kernel_exact": False, "test": None}
    (out / "history.json").write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
