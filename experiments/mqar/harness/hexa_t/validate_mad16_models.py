#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/data/mqar")
MAD = ROOT / "vendor/mad-lab"
RECIPES = ROOT / "configs/hexa_t/mad16_recipes.json"
OUTPUT = ROOT / "results/hexa_t/mad16_model_validation"
OUTPUT.mkdir(parents=True, exist_ok=True)

def load_recipes():
    return json.loads(
        RECIPES.read_text(encoding="utf-8")
    )["models"]

def run_one(model_id):
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(MAD))
    os.chdir(MAD)

    import torch
    import yaml

    from mad.registry import layer_registry
    from mad.model.language_model import LanguageModel

    recipe = next(
        item for item in load_recipes()
        if item["id"] == model_id
    )

    modules = []
    configs = []

    for layer_name in recipe["layers"]:
        entry = layer_registry[layer_name]
        config_path = MAD / entry["cfg"]
        config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

        if "dim" in config:
            config["dim"] = 128
        if "d_model" in config:
            config["d_model"] = 128
        # MAD LanguageModel requires this key in every layer config.
        config["max_length"] = 64
        if "l_max" in config:
            config["l_max"] = 64

        modules.append(entry["module"])
        configs.append(config)

    torch.manual_seed(456)
    torch.cuda.manual_seed_all(456)
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()

    model = LanguageModel(
        vocab_size=256,
        max_length=64,
        layers=modules,
        layer_cfgs=configs,
    ).cuda()

    model.train()
    tokens = torch.randint(
        0,
        256,
        (2, 64),
        device="cuda",
    )

    output = model(tokens)
    if isinstance(output, tuple):
        output = output[0]

    if output.shape[:2] != tokens.shape:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(output.shape)}"
        )

    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    result = {
        "model_id": model_id,
        "status": "ok",
        "layers": recipe["layers"],
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "parameters": parameters,
        "trainable_parameters": trainable,
        "elapsed_seconds": elapsed,
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated() / 2**20
        ),
        "gpu": torch.cuda.get_device_name(0),
    }
    print("HEXA_RESULT=" + json.dumps(result))

def parent():
    results = []
    script = Path(__file__).resolve()

    for index, recipe in enumerate(load_recipes(), start=1):
        model_id = recipe["id"]
        print(
            f"[{index:02d}/16] validating {model_id}",
            flush=True,
        )

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["WANDB_MODE"] = "disabled"

        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--one",
                    model_id,
                ],
                cwd=MAD,
                env=environment,
                capture_output=True,
                text=True,
                timeout=240,
            )

            combined = process.stdout + process.stderr
            log_path = OUTPUT / f"{model_id}.log"
            log_path.write_text(combined, encoding="utf-8")

            marker = None
            for line in process.stdout.splitlines():
                if line.startswith("HEXA_RESULT="):
                    marker = line.removeprefix("HEXA_RESULT=")

            if process.returncode == 0 and marker:
                result = json.loads(marker)
                print(
                    "  PASS "
                    f"params={result['parameters']} "
                    f"vram={result['peak_vram_mib']:.1f} MiB "
                    f"seconds={result['elapsed_seconds']:.2f}"
                )
            else:
                result = {
                    "model_id": model_id,
                    "status": "failed",
                    "returncode": process.returncode,
                    "log": str(log_path),
                    "error_tail": combined[-3000:],
                }
                print(f"  FAIL log={log_path}")

        except subprocess.TimeoutExpired as error:
            log_path = OUTPUT / f"{model_id}.log"
            text = (error.stdout or "") + (error.stderr or "")
            log_path.write_text(text, encoding="utf-8")
            result = {
                "model_id": model_id,
                "status": "timeout",
                "log": str(log_path),
            }
            print(f"  TIMEOUT log={log_path}")

        results.append(result)

        summary = {
            "models": results,
            "passed": sum(
                item["status"] == "ok"
                for item in results
            ),
            "failed": sum(
                item["status"] != "ok"
                for item in results
            ),
        }
        (OUTPUT / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

    print("\n=== MAD-16 VALIDATION SUMMARY ===")
    for result in results:
        print(
            f"{result['status'].upper():7} "
            f"{result['model_id']}"
        )

    passed = sum(
        result["status"] == "ok"
        for result in results
    )
    print(f"\nPassed: {passed}/16")
    print(f"Saved: {OUTPUT / 'summary.json'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one")
    args = parser.parse_args()

    if args.one:
        try:
            run_one(args.one)
        except Exception:
            traceback.print_exc()
            raise
    else:
        parent()

if __name__ == "__main__":
    main()
