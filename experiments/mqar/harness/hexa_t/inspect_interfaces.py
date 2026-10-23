#!/usr/bin/env python3
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/mqar")
OUTPUT = ROOT / "results/hexa_t"
OUTPUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "H1": {
        "name": "MQAR",
        "train": "train_mqar_baseline.py",
        "support": ["validate_mqar.py"],
    },
    "H2": {
        "name": "Streaming Parity",
        "train": "train_streaming_parity_transformer.py",
        "support": [
            "streaming_parity_task.py",
            "validate_parity.py",
        ],
    },
    "H3": {
        "name": "Flip-Flop",
        "train": "train_flipflop_transformer.py",
        "support": [
            "validate_flipflop.py",
            "evaluate_flipflop_stress.py",
        ],
    },
    "H4": {
        "name": "WikiText-2",
        "train": "train_wikitext2_transformer.py",
        "support": [
            "validate_wikitext2.py",
            "verify_wikitext2_checkpoint.py",
        ],
    },
    "H5": {
        "name": "Speech Commands 12-class",
        "train": "train_speechcommands_transformer.py",
        "support": [
            "validate_speechcommands12.py",
            "verify_speechcommands_checkpoint.py",
        ],
    },
}

def has_main_guard(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            text = ast.unparse(node.test)
        except Exception:
            continue
        if "__name__" in text and "__main__" in text:
            return True
    return False

def base_names(node):
    names = []
    for base in node.bases:
        try:
            names.append(ast.unparse(base))
        except Exception:
            pass
    return names

def inspect_file(relative):
    path = ROOT / relative
    result = {
        "path": relative,
        "exists": path.exists(),
    }

    if not path.exists():
        return result

    source = path.read_text(encoding="utf-8", errors="replace")
    result["sha256"] = hashlib.sha256(source.encode()).hexdigest()
    result["lines"] = source.count("\n") + 1

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        result["syntax_ok"] = False
        result["syntax_error"] = str(error)
        return result

    result["syntax_ok"] = True
    result["main_guard"] = has_main_guard(tree)
    result["classes"] = []
    result["functions"] = []
    result["config_keys"] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            result["classes"].append({
                "name": node.name,
                "bases": base_names(node),
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["functions"].append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "CONFIG"
                    and isinstance(node.value, ast.Dict)
                ):
                    keys = []
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant):
                            keys.append(str(key.value))
                    result["config_keys"] = keys

    result["model_classes"] = [
        item["name"]
        for item in result["classes"]
        if any(
            marker in base
            for base in item["bases"]
            for marker in ("Module", "LightningModule")
        )
    ]

    environment_patterns = [
        r'os\.environ\.get\(\s*["\']([^"\']+)',
        r'os\.getenv\(\s*["\']([^"\']+)',
        r'os\.environ\[\s*["\']([^"\']+)',
    ]
    environment = set()
    for pattern in environment_patterns:
        environment.update(re.findall(pattern, source))
    result["environment_variables"] = sorted(environment)

    result["uses_argparse"] = "argparse" in source
    result["cli_flags"] = sorted(set(
        re.findall(r'add_argument\(\s*["\'](--[A-Za-z0-9_-]+)', source)
    ))

    result_paths = set()
    for match in re.findall(
        r'["\']([^"\']*results/[^"\']+)["\']',
        source,
    ):
        if len(match) < 200:
            result_paths.add(match)
    result["literal_result_paths"] = sorted(result_paths)

    result["writes_summary_json"] = (
        "summary.json" in source or "summary_path" in source
    )
    result["writes_checkpoint"] = (
        "best_model.pt" in source or "checkpoint_path" in source
    )
    result["writes_history_csv"] = (
        "history.csv" in source or "history_path" in source
    )

    return result

report = {
    "framework": "HEXA-T",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "inspection_only": True,
    "tasks": {},
}

for task_id, task in TASKS.items():
    files = [task["train"], *task["support"]]
    report["tasks"][task_id] = {
        "name": task["name"],
        "train": task["train"],
        "files": [inspect_file(item) for item in files],
    }

json_path = OUTPUT / "interface_audit.json"
json_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

lines = [
    "# HEXA-T Core-5 Interface Audit",
    "",
    "| ID | Task | Train file | Main guard | Models | Env overrides | Outputs |",
    "|---|---|---|---:|---|---|---|",
]

for task_id, task in report["tasks"].items():
    train = next(
        item for item in task["files"]
        if item["path"] == task["train"]
    )
    outputs = []
    if train.get("writes_summary_json"):
        outputs.append("summary")
    if train.get("writes_checkpoint"):
        outputs.append("checkpoint")
    if train.get("writes_history_csv"):
        outputs.append("history")

    lines.append(
        f"| {task_id} | {task['name']} | `{task['train']}` | "
        f"{train.get('main_guard', False)} | "
        f"{', '.join(train.get('model_classes', [])) or '-'} | "
        f"{', '.join(train.get('environment_variables', [])) or '-'} | "
        f"{', '.join(outputs) or '-'} |"
    )

lines.extend(["", "## Detailed files", ""])

for task_id, task in report["tasks"].items():
    lines.append(f"### {task_id} — {task['name']}")
    lines.append("")
    for item in task["files"]:
        lines.append(f"- `{item['path']}`")
        lines.append(f"  - exists: {item['exists']}")
        if item["exists"]:
            lines.append(f"  - syntax: {item.get('syntax_ok')}")
            lines.append(f"  - main guard: {item.get('main_guard')}")
            lines.append(
                "  - model classes: "
                + (", ".join(item.get("model_classes", [])) or "none detected")
            )
            lines.append(
                "  - environment: "
                + (
                    ", ".join(item.get("environment_variables", []))
                    or "none"
                )
            )
            lines.append(
                "  - result paths: "
                + (
                    ", ".join(item.get("literal_result_paths", []))
                    or "none"
                )
            )
    lines.append("")

md_path = OUTPUT / "interface_audit.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("=== HEXA-T CORE-5 INTERFACES ===")
for task_id, task in report["tasks"].items():
    train = next(
        item for item in task["files"]
        if item["path"] == task["train"]
    )
    print(f"\n{task_id} {task['name']}")
    print(" train:", task["train"])
    print(" syntax:", train.get("syntax_ok"))
    print(" main_guard:", train.get("main_guard"))
    print(" models:", train.get("model_classes", []))
    print(" env:", train.get("environment_variables", []))
    print(" result_paths:", train.get("literal_result_paths", []))
    print(
        " outputs:",
        {
            "summary": train.get("writes_summary_json", False),
            "checkpoint": train.get("writes_checkpoint", False),
            "history": train.get("writes_history_csv", False),
        },
    )

print(f"\nSaved: {json_path}")
print(f"Saved: {md_path}")
