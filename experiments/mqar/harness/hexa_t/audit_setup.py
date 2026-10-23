#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/mqar")
OUTPUT = ROOT / "results/hexa_t"
OUTPUT.mkdir(parents=True, exist_ok=True)

def exact(relative):
    path = ROOT / relative
    return [relative] if path.exists() else []

def any_glob(*patterns):
    matches = set()
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_file():
                matches.add(str(path.relative_to(ROOT)))
    return sorted(matches)

checks = [
    {
        "id": "H1",
        "name": "MQAR",
        "required": True,
        "items": {
            "validation": exact("validate_mqar.py"),
            "transformer_training": exact("train_mqar_baseline.py"),
            "three_seed_summary": exact(
                "results/harness_validation/common_summary/summary.json"
            ),
            "length_sweep": exact(
                "results/harness_validation/length_sweep/length_sweep.csv"
            ),
        },
    },
    {
        "id": "H2",
        "name": "Streaming Parity",
        "required": True,
        "items": {
            "task_definition": exact("streaming_parity_task.py"),
            "transformer_training": exact(
                "train_streaming_parity_transformer.py"
            ),
            "lstm_control": exact("train_streaming_parity_lstm.py"),
            "length_sweep": exact(
                "results/harness_validation/streaming_parity/length_sweep.csv"
            ),
        },
    },
    {
        "id": "H3",
        "name": "Flip-Flop",
        "required": True,
        "items": {
            "validation": exact("validate_flipflop.py"),
            "transformer_training": exact("train_flipflop_transformer.py"),
            "lstm_control": exact("train_flipflop_lstm.py"),
            "transformer_stress_result": exact(
                "results/harness_validation/flipflop/transformer_stress.json"
            ),
            "lstm_stress_result": exact(
                "results/harness_validation/flipflop/lstm_stress.json"
            ),
        },
    },
    {
        "id": "H4",
        "name": "WikiText-2",
        "required": True,
        "items": {
            "validation": exact("validate_wikitext2.py"),
            "transformer_training": exact("train_wikitext2_transformer.py"),
            "checkpoint_verification": exact(
                "verify_wikitext2_checkpoint.py"
            ),
            "saved_result": any_glob(
                "results/**/*wikitext*/summary.json",
                "results/**/*wikitext*/*.json",
                "results/**/*wikitext*/best_model.pt",
            ),
        },
    },
    {
        "id": "H5",
        "name": "Speech Commands 12-class",
        "required": True,
        "items": {
            "data_cache": exact("cache_speechcommands12.py"),
            "validation": exact("validate_speechcommands12.py"),
            "transformer_training": exact(
                "train_speechcommands_transformer.py"
            ),
            "checkpoint_verification": exact(
                "verify_speechcommands_checkpoint.py"
            ),
            "saved_result": any_glob(
                "results/**/*speech*/summary.json",
                "results/**/*speech*/*.json",
                "results/**/*speech*/best_model.pt",
            ),
        },
    },
    {
        "id": "H6",
        "name": "CIFAR-10",
        "required": False,
        "items": {
            "dataset_pipeline": any_glob(
                "*cifar*.py", "harness/**/*cifar*.py"
            ),
            "transformer_training": any_glob(
                "*cifar*transformer*.py",
                "harness/**/*cifar*transformer*.py",
            ),
            "cnn_control": any_glob(
                "*cifar*resnet*.py",
                "harness/**/*cifar*resnet*.py",
            ),
            "saved_result": any_glob("results/**/*cifar*/*"),
        },
    },
]

for axis in checks:
    present = sum(bool(value) for value in axis["items"].values())
    total = len(axis["items"])
    axis["present"] = present
    axis["total"] = total

    if present == total:
        axis["artifact_status"] = "complete"
    elif present == 0:
        axis["artifact_status"] = "not_started"
    else:
        axis["artifact_status"] = "partial"

registry = json.loads(
    (ROOT / "configs/hexa_t/registry.json").read_text(encoding="utf-8")
)
declared = {
    axis["id"]: axis["declared_status"]
    for axis in registry["axes"]
}

report = {
    "framework": "HEXA-T",
    "audited_at": datetime.now(timezone.utc).isoformat(),
    "audit_method": "exact expected paths and task-specific result globs",
    "axes": checks,
}

json_path = OUTPUT / "setup_audit.json"
json_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

lines = [
    "# HEXA-T Setup Audit",
    "",
    "| ID | Harness | Declared | Artifacts | Checks |",
    "|---|---|---:|---:|---:|",
]

for axis in checks:
    lines.append(
        f"| {axis['id']} | {axis['name']} | "
        f"{declared[axis['id']]} | {axis['artifact_status']} | "
        f"{axis['present']}/{axis['total']} |"
    )

lines.extend(["", "## Evidence", ""])

for axis in checks:
    lines.append(f"### {axis['id']} — {axis['name']}")
    lines.append("")
    for name, evidence in axis["items"].items():
        state = "FOUND" if evidence else "MISSING"
        lines.append(f"- {state}: {name}")
        for item in evidence[:10]:
            lines.append(f"  - `{item}`")
    lines.append("")

md_path = OUTPUT / "setup_audit.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("=== CORRECTED HEXA-T AUDIT ===")
for axis in checks:
    print(
        f"{axis['id']} {axis['artifact_status'].upper():11} "
        f"{axis['present']}/{axis['total']} {axis['name']}"
    )

print(f"\nRegistry: {ROOT / 'configs/hexa_t/registry.json'}")
print(f"Audit JSON: {json_path}")
print(f"Audit Markdown: {md_path}")
print(f"Previous incorrect files backed up to: {backup}")
