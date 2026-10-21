import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_length_sweep import load_definitions
from validate_flipflop import make_dataset


SEED = 456
MAX_STEPS = 3000
PERFECT_STREAK_TARGET = 10
RESULT_DIR = Path("/data/mqar/results")
SOURCE = Path("/data/mqar/train_mqar_baseline.py")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels):
    mask = labels != -100
    predictions = logits.argmax(dim=-1)
    return ((predictions == labels) & mask).sum().item() / mask.sum().item()


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    namespace = load_definitions(SOURCE)
    config = namespace["CONFIG"]
    config.update({
        "vocab_size": 12,
        "sequence_length": 64,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 512,
    })

    inputs, labels, _, _ = make_dataset(
        num_examples=20,
        num_instructions=32,
        ignore_probability=0.80,
        seed=123,
    )
    inputs = inputs.to(device)
    labels = labels.to(device)

    model = namespace["TransformerLM"]().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    perfect_streak = 0
    final_loss = None
    final_accuracy = None

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            final_loss = loss.item()
            final_accuracy = accuracy(logits, labels)

        perfect_streak = (
            perfect_streak + 1 if final_accuracy == 1.0 else 0
        )

        if step == 1 or step % 25 == 0 or perfect_streak == 1:
            print(
                f"step={step:4d}",
                f"loss={final_loss:.6f}",
                f"accuracy={final_accuracy:.4f}",
                f"perfect_streak={perfect_streak}",
                flush=True,
            )

        if perfect_streak >= PERFECT_STREAK_TARGET:
            break

    elapsed = time.perf_counter() - started
    passed = perfect_streak >= PERFECT_STREAK_TARGET

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RESULT_DIR / "flipflop_transformer_overfit.pt"
    result_path = RESULT_DIR / "flipflop_transformer_overfit.json"

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": dict(config),
        "step": step,
        "seed": SEED,
    }, checkpoint_path)

    result = {
        "run": "flipflop_transformer_overfit",
        "passed": passed,
        "examples": 20,
        "sequence_length": 64,
        "modulus": 2,
        "steps": step,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "elapsed_seconds": elapsed,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if device.type == "cuda" else 0
        ),
        "checkpoint": str(checkpoint_path),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n==============================")
    print("FLIP-FLOP TRANSFORMER OVERFIT:", "PASS" if passed else "FAIL")
    print("steps:", step)
    print("final loss:", final_loss)
    print("final accuracy:", final_accuracy)
    print("result:", result_path)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
