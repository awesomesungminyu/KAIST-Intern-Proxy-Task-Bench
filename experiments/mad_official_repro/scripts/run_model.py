#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path("/data/mad-official-repro/repo")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark import benchmark
from mad.registry import layer_registry, model_registry

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", required=True)
parser.add_argument("--layers", nargs="+", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
logs = args.output_dir / "logs"
logs.mkdir(parents=True, exist_ok=True)

layer_modules = [
    layer_registry[name]["module"]
    for name in args.layers
]

base_layer_configs = []
for name in args.layers:
    config = yaml.safe_load(
        (ROOT / layer_registry[name]["cfg"]).read_text()
    )
    base_layer_configs.append(config)

def make_model_fn(task, vocab_size, max_length):
    configs = []

    for original in base_layer_configs:
        config = dict(original)
        config["dim"] = 128
        config["max_length"] = max_length
        configs.append(config)

    backbone = (
        "autoencoder"
        if task == "compression"
        else "language-model"
    )

    return model_registry[backbone](
        dim=128,
        vocab_size=vocab_size,
        max_length=max_length,
        layers=layer_modules,
        layer_cfgs=configs,
    )

metadata = {
    "model_id": args.model_id,
    "layers": args.layers,
    "mad_commit":
        "0f49a452b84ca0d13f8eb9c1ffa649032376fb1b",
    "data": str(ROOT / "benchmark/data"),
    "precision": "32",
    "seed": 12345,
    "difficulty_settings": 66,
    "hyperparameter_settings": {
        "learning_rate": [0.0001, 0.0005, 0.001],
        "weight_decay": [0.0, 0.1],
    },
    "expected_training_runs": 396,
    "protocol_exact": True,
    "kernel_exact": False,
    "hardware_exact": False,
    "attention_backend": "PyTorch SDPA",
    "gpu": torch.cuda.get_device_name(0),
}
(args.output_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)

scores = benchmark(
    make_model_fn=make_model_fn,
    model_id=args.model_id,
    gpus=1,
    cpus=4,
    num_trials_gpu=1,
    num_cpus_trial=2,
    data_path=str(ROOT / "benchmark/data"),
    logs_path=str(logs),
    log_to_csv=True,
    log_to_wandb=False,
    save_checkpoints=True,
    precision="32",
    persistent_workers=False,
    ray_tmp_path="/data/mad-official-repro/tmp/ray",
)

scores.rename("score").to_csv(
    args.output_dir / "mad_scores.csv"
)

summary = {
    **metadata,
    "scores": {
        str(key): float(value)
        for key, value in scores.items()
    },
    "mean_score": float(scores.mean()),
}
(args.output_dir / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

print("HEXA_RESULT=" + json.dumps(summary))
