import ast
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path("/data/mqar")
OUTPUT_DIR = ROOT / "results" / "harness_validation" / "length_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_EXAMPLES = 500
EVAL_SEED = 987654321
BATCH_SIZE = 64

CONDITIONS = [
    {"sequence_length": 64, "num_kv_pairs": 8},
    {"sequence_length": 128, "num_kv_pairs": 16},
    {"sequence_length": 256, "num_kv_pairs": 32},
]

RUNS = [
    {
        "architecture": "Transformer",
        "seed": 456,
        "source": ROOT / "train_mqar_baseline.py",
        "checkpoint": ROOT / "results/mqar_baseline/best_model.pt",
    },
    {
        "architecture": "Transformer",
        "seed": 789,
        "source": ROOT / "train_mqar_baseline.py",
        "checkpoint": ROOT / "results/harness_validation/transformer/seed_789/best_model.pt",
    },
    {
        "architecture": "Transformer",
        "seed": 101112,
        "source": ROOT / "train_mqar_baseline.py",
        "checkpoint": ROOT / "results/harness_validation/transformer/seed_101112/best_model.pt",
    },
    {
        "architecture": "Linear",
        "seed": 456,
        "source": ROOT / "train_mqar_linear_attention.py",
        "checkpoint": ROOT / "results/mqar_linear_attention/best_model.pt",
    },
    {
        "architecture": "Linear",
        "seed": 789,
        "source": ROOT / "train_mqar_linear_attention.py",
        "checkpoint": ROOT / "results/harness_validation/linear_attention/seed_789/best_model.pt",
    },
    {
        "architecture": "Linear",
        "seed": 101112,
        "source": ROOT / "train_mqar_linear_attention.py",
        "checkpoint": ROOT / "results/harness_validation/linear_attention/seed_101112/best_model.pt",
    },
    {
        "architecture": "Gated",
        "seed": 456,
        "source": ROOT / "train_mqar_gated_linear_attention.py",
        "checkpoint": ROOT / "results/mqar_gated_linear_attention/best_model.pt",
    },
    {
        "architecture": "Gated",
        "seed": 789,
        "source": ROOT / "train_mqar_gated_linear_attention.py",
        "checkpoint": ROOT / "results/harness_validation/gated_prototype/seed_789/best_model.pt",
    },
    {
        "architecture": "Gated",
        "seed": 101112,
        "source": ROOT / "train_mqar_gated_linear_attention.py",
        "checkpoint": ROOT / "results/harness_validation/gated_prototype/seed_101112/best_model.pt",
    },
]


def load_definitions(source_path):
    """훈련을 실행하지 않고 import, CONFIG, 함수, 클래스만 불러온다."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    selected = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            selected.append(node)
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "CONFIG"
                for target in node.targets
            ):
                selected.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "CONFIG":
                selected.append(node)

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "__name__": f"length_sweep_{source_path.stem}",
        "__file__": str(source_path),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


@torch.inference_mode()
def evaluate(model, loader, device):
    correct = 0
    supervised = 0

    for token_ids, labels in loader:
        token_ids = token_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(token_ids)
        mask = labels != -100
        predictions = logits.argmax(dim=-1)

        correct += ((predictions == labels) & mask).sum().item()
        supervised += mask.sum().item()

    return correct / supervised, correct, supervised


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다.")

    device = torch.device("cuda")
    namespaces = {}
    results = []

    print("device:", torch.cuda.get_device_name(0))
    print("evaluation examples per condition:", EVAL_EXAMPLES)
    print("evaluation seed:", EVAL_SEED)

    for run_index, run in enumerate(RUNS, start=1):
        source_key = str(run["source"])

        if source_key not in namespaces:
            namespaces[source_key] = load_definitions(run["source"])

        namespace = namespaces[source_key]
        checkpoint_path = run["checkpoint"]

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        train_config = dict(checkpoint["config"])

        namespace["CONFIG"].clear()
        namespace["CONFIG"].update(train_config)

        model = namespace["TransformerLM"]()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        print(
            f"\n[{run_index}/{len(RUNS)}]",
            run["architecture"],
            "seed=", run["seed"],
            "checkpoint_epoch=", checkpoint["epoch"],
        )

        for condition in CONDITIONS:
            length = condition["sequence_length"]
            pairs = condition["num_kv_pairs"]

            namespace["CONFIG"]["sequence_length"] = length
            namespace["CONFIG"]["num_kv_pairs"] = pairs

            dataset = namespace["make_dataset"](
                num_examples=EVAL_EXAMPLES,
                seed=EVAL_SEED,
            )
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
            )

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()

            try:
                accuracy, correct, supervised = evaluate(model, loader, device)
                torch.cuda.synchronize()

                elapsed = time.perf_counter() - started
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

                row = {
                    "architecture": run["architecture"],
                    "model_seed": run["seed"],
                    "checkpoint_epoch": checkpoint["epoch"],
                    "training_length": train_config["sequence_length"],
                    "training_kv_pairs": train_config["num_kv_pairs"],
                    "evaluation_length": length,
                    "evaluation_kv_pairs": pairs,
                    "evaluation_examples": EVAL_EXAMPLES,
                    "evaluation_seed": EVAL_SEED,
                    "accuracy": accuracy,
                    "correct": correct,
                    "supervised_targets": supervised,
                    "elapsed_seconds": elapsed,
                    "examples_per_second": EVAL_EXAMPLES / elapsed,
                    "peak_vram_mib": peak_vram,
                    "status": "ok",
                }

                print(
                    f"length={length:3d}",
                    f"pairs={pairs:2d}",
                    f"accuracy={accuracy:.4f}",
                    f"seconds={elapsed:.2f}",
                    f"VRAM={peak_vram:.1f} MiB",
                    flush=True,
                )

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {
                    "architecture": run["architecture"],
                    "model_seed": run["seed"],
                    "checkpoint_epoch": checkpoint["epoch"],
                    "training_length": train_config["sequence_length"],
                    "training_kv_pairs": train_config["num_kv_pairs"],
                    "evaluation_length": length,
                    "evaluation_kv_pairs": pairs,
                    "evaluation_examples": EVAL_EXAMPLES,
                    "evaluation_seed": EVAL_SEED,
                    "accuracy": None,
                    "correct": None,
                    "supervised_targets": None,
                    "elapsed_seconds": None,
                    "examples_per_second": None,
                    "peak_vram_mib": None,
                    "status": "out_of_memory",
                }
                print(f"length={length}: OUT OF MEMORY", flush=True)

            results.append(row)

        del model
        torch.cuda.empty_cache()

    csv_path = OUTPUT_DIR / "length_sweep.csv"
    json_path = OUTPUT_DIR / "length_sweep.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_path.write_text(
        json.dumps(
            {
                "evaluation_examples": EVAL_EXAMPLES,
                "evaluation_seed": EVAL_SEED,
                "conditions": CONDITIONS,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n==============================")
    print("LENGTH SWEEP COMPLETE")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()