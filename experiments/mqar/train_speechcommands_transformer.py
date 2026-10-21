import os
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/data/mqar")
CACHE_DIR = ROOT / "data" / "speechcommands12_cache"
OUTPUT_DIR = ROOT / "results" / "speechcommands_transformer_baseline"
if os.environ.get("HEXA_T_OUTPUT_DIR"):
    OUTPUT_DIR = Path(os.environ["HEXA_T_OUTPUT_DIR"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "seed": 456,
    "batch_size": 64,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "ffn_dim": 384,
    "dropout": 0.1,
    "learning_rate": 5e-4,
    "weight_decay": 0.05,
    "warmup_steps": 500,
    "max_epochs": 20,
    "early_stop_patience": 5,
    "sample_rate": 16000,
    "silence_ratio_to_known": 0.10,
}

# _HEXA_T_SMOKE_OVERRIDES_V1
_HEXA_T_INTEGER_OVERRIDES = {
    "HEXA_T_MAX_EPOCHS": "max_epochs",
    "HEXA_T_BATCH_SIZE": "batch_size",
}
for _hexa_env, _hexa_key in _HEXA_T_INTEGER_OVERRIDES.items():
    if _hexa_env in os.environ:
        CONFIG[_hexa_key] = int(os.environ[_hexa_env])

# _HEXA_T_INTEGRATION_V2
if "HEXA_T_SEED" in os.environ:
    CONFIG["seed"] = int(os.environ["HEXA_T_SEED"])

random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_cache = torch.load(
    CACHE_DIR / "training.pt",
    map_location="cpu",
)
valid_cache = torch.load(
    CACHE_DIR / "validation.pt",
    map_location="cpu",
)

LABELS = train_cache["labels"]
LABEL_TO_ID = {
    label: index for index, label in enumerate(LABELS)
}
SILENCE_ID = LABEL_TO_ID["silence"]


mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=CONFIG["sample_rate"],
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=40,
    center=False,
)


def make_silence_features(count, seed):
    noise_root = (
        ROOT
        / "data"
        / "SpeechCommands"
        / "speech_commands_v0.02"
        / "_background_noise_"
    )
    noise_paths = sorted(noise_root.glob("*.wav"))
    assert noise_paths, f"no background noise found at {noise_root}"

    noises = []
    for path in noise_paths:
        waveform, sample_rate = torchaudio.load(str(path))
        assert sample_rate == CONFIG["sample_rate"]
        noises.append(waveform[:1])

    rng = random.Random(seed)
    features = []

    with torch.no_grad():
        for index in range(count):
            noise = noises[index % len(noises)]
            maximum = noise.shape[1] - CONFIG["sample_rate"]
            offset = rng.randint(0, maximum)

            # 다양한 음량의 background noise를 silence 클래스로 사용
            amplitude = rng.uniform(0.02, 0.15)
            waveform = (
                noise[
                    :,
                    offset:offset + CONFIG["sample_rate"]
                ]
                * amplitude
            )

            feature = torch.log(mel(waveform) + 1e-6)
            feature = (
                feature - feature.mean()
            ) / (feature.std() + 1e-6)

            features.append(feature.to(torch.float16))

    return torch.stack(features)


def prepare_split(cache, seed):
    features = cache["features"]
    targets = cache["targets"]

    counts = Counter(targets.tolist())
    unknown_id = LABEL_TO_ID["unknown"]
    known_count = len(targets) - counts[unknown_id]

    silence_count = round(
        known_count * CONFIG["silence_ratio_to_known"]
    )

    silence_features = make_silence_features(
        silence_count,
        seed,
    )
    silence_targets = torch.full(
        (silence_count,),
        SILENCE_ID,
        dtype=torch.long,
    )

    return (
        torch.cat([features, silence_features], dim=0),
        torch.cat([targets, silence_targets], dim=0),
    )


print("Preparing deterministic silence samples...")
train_x, train_y = prepare_split(
    train_cache,
    CONFIG["seed"] + 1000,
)
valid_x, valid_y = prepare_split(
    valid_cache,
    CONFIG["seed"] + 2000,
)

train_generator = torch.Generator()
train_generator.manual_seed(CONFIG["seed"])

train_loader = DataLoader(
    TensorDataset(train_x, train_y),
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    generator=train_generator,
    pin_memory=True,
)
valid_loader = DataLoader(
    TensorDataset(valid_x, valid_y),
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    pin_memory=True,
)


class SequenceMixer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        output, _ = self.attention(
            x,
            x,
            x,
            need_weights=False,
        )
        return output


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(CONFIG["d_model"])
        self.mixer = SequenceMixer(
            CONFIG["d_model"],
            CONFIG["num_heads"],
            CONFIG["dropout"],
        )
        self.norm2 = nn.LayerNorm(CONFIG["d_model"])
        self.ffn = nn.Sequential(
            nn.Linear(
                CONFIG["d_model"],
                CONFIG["ffn_dim"],
            ),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(
                CONFIG["ffn_dim"],
                CONFIG["d_model"],
            ),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class KeywordTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            1,
            CONFIG["d_model"],
            kernel_size=(5, 5),
            stride=(5, 5),
        )

        # 40x98 -> 8x19 = 152 patches
        token_count = 152

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, CONFIG["d_model"])
        )
        self.position = nn.Parameter(
            torch.zeros(
                1,
                token_count + 1,
                CONFIG["d_model"],
            )
        )
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.blocks = nn.ModuleList([
            Block()
            for _ in range(CONFIG["num_layers"])
        ])
        self.norm = nn.LayerNorm(CONFIG["d_model"])
        self.head = nn.Linear(
            CONFIG["d_model"],
            len(LABELS),
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.position[:, :x.shape[1]])

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x[:, 0]))


model = KeywordTransformer().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)

total_steps = len(train_loader) * CONFIG["max_epochs"]


def lr_multiplier(step):
    if step < CONFIG["warmup_steps"]:
        return (step + 1) / CONFIG["warmup_steps"]

    progress = (
        step - CONFIG["warmup_steps"]
    ) / max(1, total_steps - CONFIG["warmup_steps"])

    return 0.5 * (
        1.0 + math.cos(math.pi * min(progress, 1.0))
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_multiplier,
)


def evaluate():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, targets in valid_loader:
            features = features.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )
            targets = targets.to(device, non_blocking=True)

            logits = model(features)
            loss = nn.functional.cross_entropy(
                logits,
                targets,
                reduction="sum",
            )

            total_loss += loss.item()
            correct += (
                logits.argmax(dim=-1) == targets
            ).sum().item()
            total += len(targets)

    return total_loss / total, correct / total


if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

print("device:", device)
print("labels:", LABELS)
print("train examples:", len(train_y))
print("validation examples:", len(valid_y))
print("train batches:", len(train_loader))
print("parameters:", sum(p.numel() for p in model.parameters()))

history = []
best_accuracy = -1.0
best_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
global_step = 0
training_start = time.perf_counter()

for epoch in range(1, CONFIG["max_epochs"] + 1):
    model.train()
    epoch_start = time.perf_counter()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for features, targets in train_loader:
        features = features.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )
        targets = targets.to(device, non_blocking=True)

        logits = model(features)
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        train_loss_sum += loss.item() * len(targets)
        train_correct += (
            logits.argmax(dim=-1) == targets
        ).sum().item()
        train_total += len(targets)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total
    valid_loss, valid_accuracy = evaluate()
    epoch_seconds = time.perf_counter() - epoch_start

    improved = valid_accuracy > best_accuracy

    if improved:
        best_accuracy = valid_accuracy
        best_loss = valid_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": CONFIG,
                "labels": LABELS,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_accuracy,
            },
            OUTPUT_DIR / "best_model.pt",
        )
    else:
        epochs_without_improvement += 1

    history.append({
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
        "seconds": epoch_seconds,
    })

    print(
        f"epoch={epoch:02d}",
        f"train_loss={train_loss:.4f}",
        f"train_acc={train_accuracy:.4f}",
        f"valid_loss={valid_loss:.4f}",
        f"valid_acc={valid_accuracy:.4f}",
        f"seconds={epoch_seconds:.1f}",
        f"best_epoch={best_epoch}",
    )

    if (
        epochs_without_improvement
        >= CONFIG["early_stop_patience"]
    ):
        print("Early stopping.")
        break

elapsed = time.perf_counter() - training_start
peak_vram = (
    torch.cuda.max_memory_allocated() / (1024 ** 2)
    if device.type == "cuda"
    else 0.0
)

history_path = OUTPUT_DIR / "history.csv"
with history_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys(),
    )
    writer.writeheader()
    writer.writerows(history)

summary = {
    "run": "speechcommands12_transformer_baseline",
    "passed": True,
    "config": CONFIG,
    "labels": LABELS,
    "best_epoch": best_epoch,
    "best_valid_loss": best_loss,
    "best_valid_accuracy": best_accuracy,
    "epochs_completed": len(history),
    "global_steps": global_step,
    "elapsed_seconds": elapsed,
    "device": str(device),
    "gpu": (
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else None
    ),
    "total_parameters": sum(
        p.numel() for p in model.parameters()
    ),
    "peak_vram_mib": peak_vram,
    "input": "1-second 16kHz normalized log-Mel",
    "feature_shape": [1, 40, 98],
    "patches": 152,
    "sequence_mixer": (
        "bidirectional standard self-attention"
    ),
    "test_split_used": False,
    "checkpoint": str(OUTPUT_DIR / "best_model.pt"),
    "history": str(history_path),
}

summary_path = OUTPUT_DIR / "summary.json"
summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("SPEECH COMMANDS BASELINE: COMPLETE")
print("best epoch:", best_epoch)
print("best validation accuracy:", best_accuracy)
print("elapsed minutes:", round(elapsed / 60, 2))
print("peak VRAM MiB:", round(peak_vram, 2))
print("checkpoint:", OUTPUT_DIR / "best_model.pt")
print("summary:", summary_path)
