import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate_length_sweep import load_definitions
from streaming_parity_task import make_dataset


ROOT = Path("/data/mqar")
OUT = ROOT / "results/harness_validation/streaming_parity"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    (
        "LSTM",
        ROOT / "train_streaming_parity_lstm.py",
        ROOT / "results/streaming_parity_lstm/best_model.pt",
    ),
    (
        "Transformer",
        ROOT / "train_streaming_parity_transformer.py",
        ROOT / "results/streaming_parity_transformer/best_model.pt",
    ),
]

LENGTHS = [32, 64, 128, 256]
EVAL_EXAMPLES = 1000
EVAL_SEED = 987654321
BATCH_SIZE = 64


@torch.inference_mode()
def evaluate(model, loader, device):
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        predictions = model(inputs).argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()

    return correct / total


def main():
    device = torch.device("cuda")
    rows = []

    for architecture, source, checkpoint_path in RUNS:
        namespace = load_definitions(source)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        namespace["CONFIG"].clear()
        namespace["CONFIG"].update(checkpoint["config"])

        model = namespace["TransformerLM"]().to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        print(f"\n=== {architecture} ===", flush=True)

        for length in LENGTHS:
            inputs, labels = make_dataset(
                num_examples=EVAL_EXAMPLES,
                sequence_length=length,
                modulus=2,
                seed=EVAL_SEED,
            )
            loader = DataLoader(
                TensorDataset(inputs, labels),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            accuracy = evaluate(model, loader, device)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            row = {
                "architecture": architecture,
                "training_length": 32,
                "evaluation_length": length,
                "evaluation_examples": EVAL_EXAMPLES,
                "evaluation_seed": EVAL_SEED,
                "accuracy": accuracy,
                "elapsed_seconds": elapsed,
                "peak_vram_mib": (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                ),
            }
            rows.append(row)

            print(
                f"length={length:3d}",
                f"accuracy={accuracy:.4f}",
                f"seconds={elapsed:.2f}",
                flush=True,
            )

        del model
        torch.cuda.empty_cache()

    csv_path = OUT / "length_sweep.csv"
    json_path = OUT / "length_sweep.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print("\n==============================")
    print("STREAMING PARITY LENGTH SWEEP COMPLETE")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
