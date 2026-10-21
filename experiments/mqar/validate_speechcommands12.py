import json
import random
from collections import Counter
from pathlib import Path

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

ROOT = Path("/data/mqar")
DATA_ROOT = ROOT / "data"
RESULT = ROOT / "results" / "speechcommands12_data_validation.json"

WANTED = {
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",
}
SAMPLE_RATE = 16000
SEED = 456


class SC12(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__(
            root=str(DATA_ROOT),
            url="speech_commands_v0.02",
            download=False,
        )

        base = Path(self._path)

        def read_list(name):
            return {
                str(base / line.strip())
                for line in (base / name).read_text().splitlines()
                if line.strip()
            }

        valid = read_list("validation_list.txt")
        test = read_list("testing_list.txt")

        if subset == "validation":
            walker = valid
        elif subset == "testing":
            walker = test
        elif subset == "training":
            walker = set(self._walker) - valid - test
        else:
            raise ValueError(subset)

        known = sorted(
            path for path in walker
            if Path(path).parent.name in WANTED
        )
        unknown = sorted(
            path for path in walker
            if Path(path).parent.name not in WANTED
        )

        rng = random.Random(SEED + {
            "training": 0,
            "validation": 1,
            "testing": 2,
        }[subset])

        # Google Speech Commands의 일반적인 10% unknown 조건
        unknown_count = min(len(unknown), round(len(known) * 0.10))
        selected_unknown = rng.sample(unknown, unknown_count)

        self._walker = known + selected_unknown
        self._known_paths = set(known)

    def __getitem__(self, index):
        waveform, sr, label, speaker, utterance = super().__getitem__(index)

        if waveform.shape[1] < SAMPLE_RATE:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, SAMPLE_RATE - waveform.shape[1]),
            )
        else:
            waveform = waveform[:, :SAMPLE_RATE]

        target = label if label in WANTED else "unknown"
        return waveform, sr, target, speaker, utterance


mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=40,
    center=False,
)

splits = {}
label_names = sorted(WANTED) + ["unknown", "silence"]

for split in ("training", "validation", "testing"):
    dataset = SC12(split)
    counts = Counter()

    check_count = min(256, len(dataset))
    indices = torch.linspace(
        0, len(dataset) - 1, check_count
    ).long().tolist()

    feature_shape = None

    for index in indices:
        waveform, sr, label, _, _ = dataset[index]
        assert sr == SAMPLE_RATE
        assert waveform.shape == (1, SAMPLE_RATE)

        features = torch.log(mel(waveform) + 1e-6)
        feature_shape = list(features.shape)

        assert torch.isfinite(features).all()
        assert features.shape[1] == 40
        counts[label] += 1

    full_counts = Counter(
        Path(path).parent.name
        if Path(path).parent.name in WANTED
        else "unknown"
        for path in dataset._walker
    )

    splits[split] = {
        "examples_without_silence": len(dataset),
        "class_counts": dict(sorted(full_counts.items())),
        "checked_examples": check_count,
        "waveform_shape": [1, SAMPLE_RATE],
        "log_mel_shape": feature_shape,
    }

    print(
        f"{split:10s}",
        f"examples={len(dataset):,}",
        f"log_mel={feature_shape}",
        "PASS",
    )

RESULT.parent.mkdir(parents=True, exist_ok=True)
RESULT.write_text(
    json.dumps(
        {
            "run": "speechcommands12_preprocessing_validation",
            "passed": True,
            "wanted_commands": sorted(WANTED),
            "labels": label_names,
            "unknown_ratio_to_known": 0.10,
            "silence_policy": (
                "Add deterministic background-noise silence samples "
                "during the training harness."
            ),
            "sample_rate": SAMPLE_RATE,
            "waveform_samples": SAMPLE_RATE,
            "feature": "log-Mel spectrogram",
            "n_mels": 40,
            "n_fft": 400,
            "hop_length": 160,
            "splits": splits,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("=" * 56)
print("SPEECH COMMANDS 12-CLASS PREPROCESSING: PASS")
print("saved:", RESULT)
