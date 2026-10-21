#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEGACY_RUNS = [
    ("Transformer", 456, "results/mqar_baseline/summary.json"),
    ("Transformer", 789, "results/harness_validation/transformer/seed_789/summary.json"),
    ("Transformer", 101112, "results/harness_validation/transformer/seed_101112/summary.json"),
("GQA-2 Transformer", 456, "results/harness_validation/gqa2/seed_456/summary.json"),
("GQA-2 Transformer", 789, "results/harness_validation/gqa2/seed_789/summary.json"),
("GQA-2 Transformer", 101112, "results/harness_validation/gqa2/seed_101112/summary.json"),
    ("Linear attention", 456, "results/mqar_linear_attention/summary.json"),
    ("Linear attention", 789, "results/harness_validation/linear_attention/seed_789/summary.json"),
    ("Linear attention", 101112, "results/harness_validation/linear_attention/seed_101112/summary.json"),
    ("Gated prototype", 456, "results/mqar_gated_linear_attention/summary.json"),
    ("Gated prototype", 789, "results/harness_validation/gated_prototype/seed_789/summary.json"),
    ("Gated prototype", 101112, "results/harness_validation/gated_prototype/seed_101112/summary.json"),
]

METRIC_KEYS = [
    "passed",
    "best_epoch",
    "best_valid_loss",
    "best_valid_accuracy",
    "epochs_completed",
    "global_steps",
    "elapsed_seconds",
    "steps_per_second",
    "total_parameters",
    "peak_vram_mib",
    "device",
    "gpu",
    "torch_version",
    "torch_cuda_version",
    "dtype",
    "position_encoding",
    "sequence_mixer",
    "theoretical_complexity",
    "checkpoint",
    "history",
]


def make_record(architecture, expected_seed, summary_path, job_status=None):
    record = {
        "architecture": architecture,
        "seed": expected_seed,
        "status": "missing",
        "returncode": None,
        "source_summary": str(summary_path),
    }

    if job_status:
        record["status"] = job_status.get("status", "unknown")
        record["returncode"] = job_status.get("returncode")

    if not summary_path.is_file():
        return record

    data = json.loads(summary_path.read_text())
    config = data.get("config", {})

    record["seed"] = config.get("model_seed", expected_seed)
    record["status"] = (
        job_status.get("status", "ok")
        if job_status
        else "ok"
    )

    for key in METRIC_KEYS:
        value = data.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            record[key] = value

    accuracy = record.get("best_valid_accuracy")
    record["best_valid_accuracy_percent"] = (
        accuracy * 100
        if isinstance(accuracy, (int, float))
        else None
    )

    return record


def collect_legacy():
    rows = []
    for architecture, seed, relative_path in LEGACY_RUNS:
        rows.append(
            make_record(
                architecture,
                seed,
                ROOT / relative_path,
            )
        )
    return rows


def collect_harness_run(run_dir):
    rows = []

    for job_dir in sorted(run_dir.iterdir()):
        if not job_dir.is_dir():
            continue

        status_path = job_dir / "status.json"
        if not status_path.is_file():
            continue

        job_status = json.loads(status_path.read_text())
        summary_path = job_dir / "artifacts" / "summary.json"

        rows.append(
            make_record(
                job_status.get("architecture", "unknown"),
                job_status.get("seed"),
                summary_path,
                job_status,
            )
        )

    return rows


def group_summary(rows):
    grouped = {}

    for row in rows:
        if row.get("status") != "ok":
            continue

        architecture = row["architecture"]
        grouped.setdefault(architecture, []).append(row)

    output = []

    for architecture, items in sorted(grouped.items()):
        accuracies = [
            item["best_valid_accuracy"]
            for item in items
            if isinstance(item.get("best_valid_accuracy"), (int, float))
        ]
        losses = [
            item["best_valid_loss"]
            for item in items
            if isinstance(item.get("best_valid_loss"), (int, float))
        ]

        output.append({
            "architecture": architecture,
            "run_count": len(items),
            "seeds": sorted(item["seed"] for item in items),
            "mean_valid_accuracy_percent": (
                statistics.mean(accuracies) * 100
                if accuracies else None
            ),
            "std_valid_accuracy_percent": (
                statistics.stdev(accuracies) * 100
                if len(accuracies) >= 2 else 0.0
            ),
            "mean_valid_loss": (
                statistics.mean(losses)
                if losses else None
            ),
        })

    return output


def write_csv(path, rows):
    preferred = [
        "architecture",
        "run_count",
        "seeds",
        "mean_valid_accuracy_percent",
        "std_valid_accuracy_percent",
        "min_valid_accuracy_percent",
        "max_valid_accuracy_percent",
        "seed",
        "status",
        "returncode",
        "passed",
        "best_epoch",
        "best_valid_loss",
        "best_valid_accuracy",
        "best_valid_accuracy_percent",
        "epochs_completed",
        "global_steps",
        "elapsed_seconds",
        "steps_per_second",
        "total_parameters",
        "peak_vram_mib",
        "device",
        "gpu",
        "torch_version",
        "torch_cuda_version",
        "dtype",
        "position_encoding",
        "sequence_mixer",
        "theoretical_complexity",
        "checkpoint",
        "history",
        "source_summary",
    ]

    fieldnames = [
        key for key in preferred
        if any(key in row for row in rows)
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key)
                for key in fieldnames
            })


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--legacy", action="store_true")
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.legacy:
        rows = collect_legacy()
        default_output = (
            ROOT
            / "results"
            / "harness_validation"
            / "common_summary"
        )
    else:
        run_dir = args.run_dir.resolve()
        rows = collect_harness_run(run_dir)
        default_output = run_dir

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else default_output
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_summary(rows)

    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "by_architecture.csv", groups)

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": rows,
                "by_architecture": groups,
            },
            indent=2,
        )
        + "\n"
    )

    print("INDIVIDUAL RUNS")
    for row in rows:
        print(
            row["architecture"],
            f"seed={row['seed']}",
            f"status={row['status']}",
            f"accuracy={row.get('best_valid_accuracy_percent')}",
        )

    print("\nBY ARCHITECTURE")
    for group in groups:
        print(
            group["architecture"],
            f"runs={group['run_count']}",
            f"seeds={group['seeds']}",
            f"accuracy={group['mean_valid_accuracy_percent']:.4f}"
            f"±{group['std_valid_accuracy_percent']:.4f}%",
        )

    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
