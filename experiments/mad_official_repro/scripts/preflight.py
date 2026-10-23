#!/usr/bin/env python3
import gc
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/data/mad-official-repro/repo")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mad.configs import make_benchmark_mad_configs
from mad.registry import layer_registry, model_registry

configs = make_benchmark_mad_configs(
    data_path=str(ROOT / "benchmark/data"),
    precision="32",
    persistent_data_workers=False,
)
datasets = {config.dataset_path for config in configs}
missing = [
    path for path in datasets
    if not Path(path, "train").is_dir()
    or not Path(path, "test").is_dir()
]

print("Training runs per model:", len(configs))
print("Unique datasets:", len(datasets))
print("Missing datasets:", len(missing))

if len(configs) != 396:
    raise RuntimeError("Expected 396 runs per model")
if len(datasets) != 66:
    raise RuntimeError("Expected 66 difficulty settings")
if missing:
    raise RuntimeError(f"Missing datasets: {missing[:3]}")

models = {
    "mh_attention_swiglu": [
        "mh-attention", "swiglu",
        "mh-attention", "swiglu",
    ],
    "striped_mh_hyena_moe": [
        "mh-hyena", "moe-mlp",
        "mh-attention", "moe-mlp",
    ],
}

for model_id, layer_names in models.items():
    for backbone in ("language-model", "autoencoder"):
        modules = []
        layer_configs = []

        for layer_name in layer_names:
            entry = layer_registry[layer_name]
            config = yaml.safe_load(
                (ROOT / entry["cfg"]).read_text()
            )
            config["dim"] = 128
            config["max_length"] = 64
            modules.append(entry["module"])
            layer_configs.append(config)

        model = model_registry[backbone](
            dim=128,
            vocab_size=32,
            max_length=64,
            layers=modules,
            layer_cfgs=layer_configs,
        ).cuda()

        tokens = torch.randint(
            0, 32, (2, 64), device="cuda"
        )
        output = model(tokens)
        if isinstance(output, tuple):
            output = output[0]

        output.float().square().mean().backward()
        torch.cuda.synchronize()

        if not torch.isfinite(output).all():
            raise RuntimeError(
                f"{model_id}/{backbone}: non-finite output"
            )

        print(
            "PASS",
            model_id,
            backbone,
            tuple(output.shape),
        )

        del model, tokens, output
        gc.collect()
        torch.cuda.empty_cache()

print("PASS: OFFICIAL MAD DATA AND BOTH MODELS ARE READY")
