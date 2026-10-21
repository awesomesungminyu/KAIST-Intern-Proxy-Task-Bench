#!/usr/bin/env python3
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/mqar")
sys.path.insert(0, str(ROOT / "harness" / "hexa_t"))

import run_mad16_hexat as runner

MODELS = [
    "mh_attention_swiglu",
    "striped_mh_hyena_moe",
]

runner.RESULTS = (
    ROOT / "results" / "hexa_t" / "mad2_h1_comparison_repeat1"
)
runner.TASKS = {
    "H1": ROOT / "train_mqar_mad.py",
    "H2": ROOT / "train_streaming_parity_mad.py",
    "H3": ROOT / "train_flipflop_mad.py",
    "H4": ROOT / "train_wikitext2_mad.py",
}
runner.load_models = lambda: MODELS

manifest_path = runner.RESULTS / "h1_h4_manifest.json"
runner.RESULTS.mkdir(parents=True, exist_ok=True)

manifest = {
    "experiment": "MAD two-model HEXA-T H1-H4 reproducibility repeat 1",
    "models": MODELS,
    "tasks": list(runner.TASKS),
    "seeds": list(runner.SEEDS),
    "expected_runs": 24,
    "attention_backend": "torch-sdpa-turing-compatibility",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "status": "running",
}
runner.write_json(manifest_path, manifest)

failures = runner.run_phase("full")
rows = []

for model_id in MODELS:
    for task in runner.TASKS:
        for seed in runner.SEEDS:
            summary_path = (
                runner.RESULTS
                / "full"
                / model_id
                / task
                / f"seed_{seed}"
                / "summary.json"
            )

            if not summary_path.exists():
                continue

            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

            accuracy = summary.get("best_valid_accuracy")
            rows.append({
                "model_id": model_id,
                "task": task,
                "seed": seed,
                "accuracy_percent": (
                    float(accuracy) * 100
                    if accuracy is not None
                    else None
                ),
                "valid_loss": summary.get("best_valid_loss"),
                "valid_ppl": summary.get("best_valid_ppl"),
                "best_epoch": summary.get("best_epoch"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "peak_vram_mib": summary.get("peak_vram_mib"),
                "total_parameters":
                    summary.get("total_parameters"),
                "summary": str(summary_path),
            })

csv_path = runner.RESULTS / "h1_h4_runs.csv"
fields = [
    "model_id",
    "task",
    "seed",
    "accuracy_percent",
    "valid_loss",
    "valid_ppl",
    "best_epoch",
    "elapsed_seconds",
    "peak_vram_mib",
    "total_parameters",
    "summary",
]

with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

aggregates = []

for model_id in MODELS:
    for task in runner.TASKS:
        group = [
            row for row in rows
            if row["model_id"] == model_id
            and row["task"] == task
        ]

        accuracies = [
            row["accuracy_percent"]
            for row in group
            if row["accuracy_percent"] is not None
        ]
        ppls = [
            float(row["valid_ppl"])
            for row in group
            if row["valid_ppl"] is not None
        ]
        elapsed = [
            float(row["elapsed_seconds"])
            for row in group
            if row["elapsed_seconds"] is not None
        ]
        vram = [
            float(row["peak_vram_mib"])
            for row in group
            if row["peak_vram_mib"] is not None
        ]

        aggregate = {
            "model_id": model_id,
            "task": task,
            "runs": len(group),
            "accuracy_mean_percent": (
                statistics.mean(accuracies)
                if accuracies else None
            ),
            "accuracy_std_percent": (
                statistics.stdev(accuracies)
                if len(accuracies) > 1 else None
            ),
            "ppl_mean": (
                statistics.mean(ppls) if ppls else None
            ),
            "ppl_std": (
                statistics.stdev(ppls)
                if len(ppls) > 1 else None
            ),
            "elapsed_mean_seconds": (
                statistics.mean(elapsed) if elapsed else None
            ),
            "peak_vram_mean_mib": (
                statistics.mean(vram) if vram else None
            ),
        }
        aggregates.append(aggregate)

        if accuracies:
            print(
                f"{model_id} {task}: "
                f"accuracy={statistics.mean(accuracies):.4f}"
                f"±{statistics.stdev(accuracies):.4f}%"
            )
        elif ppls:
            print(
                f"{model_id} {task}: "
                f"PPL={statistics.mean(ppls):.4f}"
                f"±{statistics.stdev(ppls):.4f}"
            )

aggregate_path = runner.RESULTS / "h1_h4_by_model_task.json"
runner.write_json(aggregate_path, aggregates)

manifest.update({
    "status": "ok" if not failures else "completed_with_failures",
    "completed_runs": len(rows),
    "failures": failures,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "runs_csv": str(csv_path),
    "aggregate_json": str(aggregate_path),
})
runner.write_json(manifest_path, manifest)

print(f"Completed: {len(rows)}/24")
print(f"CSV: {csv_path}")
print(f"Summary: {aggregate_path}")
raise SystemExit(0 if not failures else 1)
