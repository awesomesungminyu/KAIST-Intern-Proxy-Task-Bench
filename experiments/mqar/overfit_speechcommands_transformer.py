import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

ROOT = Path("/data/mqar")
DATA_ROOT = ROOT / "data"
RESULT_DIR = ROOT / "results"
SEED = 456
SAMPLE_RATE = 16000

COMMANDS = sorted({
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",
})
LABELS = COMMANDS + ["unknown", "silence"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingAudio(SPEECHCOMMANDS):
    def __init__(self):
        super().__init__(
            root=str(DATA_ROOT),
            url="speech_commands_v0.02",
            download=False,
        )

        base = Path(self._path)

        validation = {
            str(base / line.strip())
            for line in (base / "validation_list.txt").read_text().splitlines()
            if line.strip()
        }
        testing = {
            str(base / line.strip())
            for line in (base / "testing_list.txt").read_text().splitlines()
            if line.strip()
        }

        self._walker = sorted(
            path for path in self._walker
            if path not in validation and path not in testing
        )


audio = TrainingAudio()
base = Path(audio._path)

paths_by_label = {}
for path in audio._walker:
    paths_by_label.setdefault(Path(path).parent.name, []).append(path)

unknown_paths = [
    path
    for label, paths in paths_by_label.items()
    if label not in COMMANDS
    for path in paths
]

rng = random.Random(SEED)
selected = []

# 각 명령어 10개
for label in COMMANDS:
    for path in rng.sample(paths_by_label[label], 10):
        selected.append((path, label))

# unknown 10개
for path in rng.sample(unknown_paths, 10):
    selected.append((path, "unknown"))

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=40,
    center=False,
)

features = []
targets = []

for path, label in selected:
    waveform, sample_rate = torchaudio.load(path)
    assert sample_rate == SAMPLE_RATE

    waveform = waveform[:1]

    if waveform.shape[1] < SAMPLE_RATE:
        waveform = nn.functional.pad(
            waveform,
            (0, SAMPLE_RATE - waveform.shape[1]),
        )
    else:
        waveform = waveform[:, :SAMPLE_RATE]

    feature = torch.log(mel(waveform) + 1e-6)
    feature = (feature - feature.mean()) / (feature.std() + 1e-6)

    features.append(feature)
    targets.append(LABEL_TO_ID[label])

# 실제 background noise에서 silence 10개 생성
noise_paths = sorted((base / "_background_noise_").glob("*.wav"))
assert noise_paths, "background-noise files were not found"

for index in range(10):
    noise, sample_rate = torchaudio.load(str(noise_paths[index % len(noise_paths)]))
    assert sample_rate == SAMPLE_RATE

    noise = noise[:1]
    max_offset = noise.shape[1] - SAMPLE_RATE
    offset = rng.randint(0, max_offset)
    waveform = noise[:, offset:offset + SAMPLE_RATE] * 0.1

    feature = torch.log(mel(waveform) + 1e-6)
    feature = (feature - feature.mean()) / (feature.std() + 1e-6)

    features.append(feature)
    targets.append(LABEL_TO_ID["silence"])

x = torch.stack(features).to(device)
y = torch.tensor(targets, dtype=torch.long, device=device)

permutation = torch.randperm(len(y), device=device)
x = x[permutation]
y = y[permutation]


class SequenceMixer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, x):
        output, _ = self.attention(
            x, x, x,
            need_weights=False,
        )
        return output


class Block(nn.Module):
    def __init__(self, d_model=128, num_heads=4, ffn_dim=384):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = SequenceMixer(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
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
            128,
            kernel_size=(5, 5),
            stride=(5, 5),
        )

        # 40x98 -> 8x19 = 152 patches, CLS 포함 153
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.position = nn.Parameter(torch.zeros(1, 153, 128))

        self.blocks = nn.ModuleList([
            Block() for _ in range(2)
        ])

        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, len(LABELS))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position[:, :x.shape[1]]

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x[:, 0]))


model = KeywordTransformer().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()
perfect_streak = 0
passed = False
final_loss = None
final_accuracy = None

print("device:", device)
print("examples:", len(y))
print("classes:", len(LABELS))
print("input shape:", tuple(x.shape))
print("parameters:", sum(p.numel() for p in model.parameters()))

for step in range(1, 501):
    model.train()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        accuracy = (logits.argmax(dim=-1) == y).float().mean().item()

    final_loss = loss.item()
    final_accuracy = accuracy

    if accuracy == 1.0:
        perfect_streak += 1
    else:
        perfect_streak = 0

    if step == 1 or step % 25 == 0 or perfect_streak == 1:
        print(
            f"step={step:3d}",
            f"loss={final_loss:.6f}",
            f"accuracy={final_accuracy:.4f}",
            f"perfect_streak={perfect_streak}",
        )

    if perfect_streak >= 5:
        passed = True
        break

elapsed = time.perf_counter() - start
peak_vram = (
    torch.cuda.max_memory_allocated() / (1024 ** 2)
    if device.type == "cuda"
    else 0.0
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_path = RESULT_DIR / "speechcommands_transformer_overfit.pt"
result_path = RESULT_DIR / "speechcommands_transformer_overfit.json"

torch.save(model.state_dict(), checkpoint_path)

result = {
    "run": "speechcommands12_transformer_overfit",
    "passed": passed,
    "steps": step,
    "examples": len(y),
    "examples_per_class": 10,
    "classes": LABELS,
    "input": "1-second 16kHz log-Mel spectrogram",
    "input_shape": list(x.shape),
    "patching": "Conv2d kernel=5x5 stride=5x5",
    "sequence_mixer": "bidirectional standard self-attention",
    "final_loss": final_loss,
    "final_accuracy": final_accuracy,
    "elapsed_seconds": elapsed,
    "peak_vram_mib": peak_vram,
    "total_parameters": sum(p.numel() for p in model.parameters()),
    "checkpoint": str(checkpoint_path),
}

result_path.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("=" * 48)
print(
    "SPEECH COMMANDS TRANSFORMER OVERFIT:",
    "PASS" if passed else "FAIL",
)
print("steps:", step)
print("final loss:", final_loss)
print("final accuracy:", final_accuracy)
print("elapsed seconds:", round(elapsed, 3))
print("peak VRAM MiB:", round(peak_vram, 2))
print("result:", result_path)
