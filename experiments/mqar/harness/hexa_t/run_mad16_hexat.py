#!/usr/bin/env python3
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/mqar")
MAD = ROOT / "vendor/mad-lab"
PYTHON = Path("/data/envs/hexa-mad/bin/python")
RESULTS = ROOT / "results/hexa_t/mad16_hexat_v1"

TASKS = {
    "H1": ROOT / "train_mqar_mad.py",
    "H2": ROOT / "train_streaming_parity_mad.py",
    "H3": ROOT / "train_flipflop_mad.py",
    "H4": ROOT / "train_wikitext2_mad.py",
}
SEEDS = (456, 789, 101112)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_models():
    path = ROOT / "configs/hexa_t/mad16_recipes.json"
    return [
        item["id"]
        for item in json.loads(
            path.read_text(encoding="utf-8")
        )["models"]
    ]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def valid_summary(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(value, dict) and bool(value)
    except Exception:
        return False


def make_environment(model, task, seed, output, smoke):
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")

    env.update({
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": (
            str(MAD)
            if not old_pythonpath
            else str(MAD) + os.pathsep + old_pythonpath
        ),
        "HEXA_T_MAD_MODEL_ID": model,
        "HEXA_T_TASK": task,
        "HEXA_T_SEED": str(seed),
        "HARNESS_SEED": str(seed),
        "HEXA_T_OUTPUT_DIR": str(output),
        "HARNESS_OUTPUT_DIR": str(output),
    })

    if smoke:
        overrides = {
            "TRAIN_EXAMPLES": "512",
            "VALID_EXAMPLES": "256",
            "BATCH_SIZE": "64",
            "MAX_EPOCHS": "1",
        }
        for prefix in ("HEXA_T", "HARNESS"):
            for name, value in overrides.items():
                env[f"{prefix}_{name}"] = value

    return env


def run_one(phase, model, task, seed):
    output = (
        RESULTS / phase / model / task / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    summary = output / "summary.json"
    log = output / "run.log"

    if valid_summary(summary):
        print(
            f"SKIP {phase} {model} {task} seed={seed}",
            flush=True,
        )
        return True

    print(
        f"RUN  {phase} {model} {task} seed={seed}",
        flush=True,
    )

    started = time.perf_counter()

    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [str(PYTHON), str(TASKS[task])],
            cwd=ROOT,
            env=make_environment(
                model,
                task,
                seed,
                output,
                phase == "smoke",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()

        process.wait()

    elapsed = time.perf_counter() - started
    ok = process.returncode == 0 and valid_summary(summary)

    write_json(
        output / "run_status.json",
        {
            "phase": phase,
            "model": model,
            "task": task,
            "seed": seed,
            "status": "ok" if ok else "failed",
            "returncode": process.returncode,
            "elapsed_seconds": elapsed,
            "log": str(log),
            "summary": str(summary),
            "finished_at": now(),
        },
    )

    print(
        f"{'OK' if ok else 'FAIL'} "
        f"{phase} {model} {task} seed={seed} "
        f"seconds={elapsed:.1f} log={log}",
        flush=True,
    )
    return ok


def run_phase(phase):
    seeds = (456,) if phase == "smoke" else SEEDS
    failures = []
    completed = 0
    total = len(load_models()) * len(TASKS) * len(seeds)

    for model in load_models():
        for task in TASKS:
            for seed in seeds:
                try:
                    ok = run_one(
                        phase, model, task, seed
                    )
                except Exception as error:
                    ok = False
                    print(
                        f"FAIL {phase} {model} {task} "
                        f"seed={seed}: {error!r}",
                        flush=True,
                    )

                completed += 1
                if not ok:
                    failures.append({
                        "model_id": model,
                        "task": task,
                        "seed": seed,
                    })

                write_json(
                    RESULTS / f"{phase}_progress.json",
                    {
                        "phase": phase,
                        "completed": completed,
                        "total": total,
                        "failed": failures,
                        "updated_at": now(),
                    },
                )

    return failures
