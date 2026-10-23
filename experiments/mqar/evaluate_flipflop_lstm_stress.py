import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate_length_sweep import load_definitions
from validate_flipflop import make_dataset, WRITE, READ


ROOT = Path("/data/mqar")
SOURCE = ROOT / "train_flipflop_lstm.py"
CHECKPOINT = ROOT / "results/flipflop_lstm/best_model.pt"
OUTPUT = ROOT / "results/harness_validation/flipflop"
OUTPUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("id", 64, 0.80, 10000, 64),
    ("dense_ood", 64, 0.10, 10000, 64),
    ("sparse_ood", 64, 0.98, 10000, 64),
    ("long_sparse_ood", 512, 0.98, 2000, 8),
]

BINS = [
    (0, 4, "0-4"),
    (5, 15, "5-15"),
    (16, 31, "16-31"),
    (32, 63, "32-63"),
    (64, 127, "64-127"),
    (128, 10**9, "128+"),
]


def distance_bin(distance):
    for low, high, name in BINS:
        if low <= distance <= high:
            return name
    raise RuntimeError(distance)


@torch.inference_mode()
def evaluate(model, loader, device):
    totals = defaultdict(int)
    correct = defaultdict(int)

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        predictions = model(inputs).argmax(dim=-1)

        inputs_cpu = inputs.cpu()
        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()

        for row in range(inputs_cpu.size(0)):
            last_write = 0
            instructions = inputs_cpu.size(1) // 2

            for step in range(instructions):
                position = 2 * step
                operation = inputs_cpu[row, position].item()

                if operation == WRITE:
                    last_write = step
                elif operation == READ:
                    distance = step - last_write
                    name = distance_bin(distance)
                    is_correct = (
                        predictions_cpu[row, position].item()
                        == labels_cpu[row, position].item()
                    )

                    totals["all"] += 1
                    totals[name] += 1
                    correct["all"] += int(is_correct)
                    correct[name] += int(is_correct)

    return totals, correct


def main():
    device = torch.device("cuda")
    namespace = load_definitions(SOURCE)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    namespace["CONFIG"].clear()
    namespace["CONFIG"].update(checkpoint["config"])

    model = namespace["TransformerLM"]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    results = []

    for name, instructions, ignore_probability, examples, batch_size in CONDITIONS:
        inputs, labels, _, _ = make_dataset(
            num_examples=examples,
            num_instructions=instructions,
            ignore_probability=ignore_probability,
            seed=987654321,
        )
        loader = DataLoader(
            TensorDataset(inputs, labels),
            batch_size=batch_size,
            shuffle=False,
        )

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        totals, correct = evaluate(model, loader, device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        accuracy = correct["all"] / totals["all"]
        errors = totals["all"] - correct["all"]

        result = {
            "condition": name,
            "instructions": instructions,
            "tokens": instructions * 2,
            "ignore_probability": ignore_probability,
            "examples": examples,
            "read_targets": totals["all"],
            "correct": correct["all"],
            "errors": errors,
            "accuracy": accuracy,
            "glitch_rate": 1.0 - accuracy,
            "elapsed_seconds": elapsed,
            "distance_bins": {},
        }

        print(
            f"\n{name}",
            f"reads={totals['all']}",
            f"errors={errors}",
            f"accuracy={accuracy:.6f}",
        )

        for _, _, bin_name in BINS:
            count = totals[bin_name]
            if count:
                bin_accuracy = correct[bin_name] / count
                result["distance_bins"][bin_name] = {
                    "count": count,
                    "accuracy": bin_accuracy,
                    "errors": count - correct[bin_name],
                }
                print(
                    f"  distance={bin_name:6s}",
                    f"count={count:6d}",
                    f"accuracy={bin_accuracy:.6f}",
                )

        results.append(result)

    output_path = OUTPUT / "lstm_stress.json"
    output_path.write_text(
        json.dumps({
            "run": "flipflop_lstm_stress",
            "checkpoint": str(CHECKPOINT),
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n==============================")
    print("FLIP-FLOP STRESS EVALUATION COMPLETE")
    print("saved:", output_path)


if __name__ == "__main__":
    main()
