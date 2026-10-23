import json
from pathlib import Path

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

ROOT = Path("/data/mqar")
DATA_ROOT = ROOT / "data"
RESULT_PATH = ROOT / "results" / "speechcommands_data_validation.json"

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CLASSES = 35


class SpeechCommandsSubset(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__(
            root=str(DATA_ROOT),
            url="speech_commands_v0.02",
            download=True,
        )

        base = Path(self._path)

        def load_list(filename):
            with open(base / filename, encoding="utf-8") as file:
                return {
                    str(base / line.strip())
                    for line in file
                    if line.strip()
                }

        validation = load_list("validation_list.txt")
        testing = load_list("testing_list.txt")

        if subset == "validation":
            self._walker = sorted(validation)
        elif subset == "testing":
            self._walker = sorted(testing)
        elif subset == "training":
            self._walker = sorted(
                path
                for path in self._walker
                if path not in validation and path not in testing
            )
        else:
            raise ValueError(subset)


print("Loading Speech Commands v0.02...")
datasets = {
    name: SpeechCommandsSubset(name)
    for name in ("training", "validation", "testing")
}

path_sets = {
    name: set(dataset._walker)
    for name, dataset in datasets.items()
}

assert path_sets["training"].isdisjoint(path_sets["validation"])
assert path_sets["training"].isdisjoint(path_sets["testing"])
assert path_sets["validation"].isdisjoint(path_sets["testing"])
print("official split separation: PASS")

all_labels = set()
split_results = {}

for split_name, dataset in datasets.items():
    labels = set()
    speakers = set()
    lengths = []
    checked = min(len(dataset), 256)

    sample_indices = torch.linspace(
        0,
        len(dataset) - 1,
        steps=checked,
    ).long().tolist()

    for index in sample_indices:
        waveform, sample_rate, label, speaker_id, utterance_number = dataset[index]

        assert sample_rate == EXPECTED_SAMPLE_RATE
        assert waveform.ndim == 2
        assert waveform.shape[0] == 1
        assert waveform.shape[1] > 0
        assert waveform.shape[1] <= EXPECTED_SAMPLE_RATE

        labels.add(label)
        speakers.add(speaker_id)
        lengths.append(waveform.shape[1])

    path_labels = {
        Path(path).parent.name
        for path in dataset._walker
    }
    all_labels.update(path_labels)

    split_results[split_name] = {
        "examples": len(dataset),
        "classes": len(path_labels),
        "sample_rate": EXPECTED_SAMPLE_RATE,
        "checked_waveforms": checked,
        "checked_min_samples": min(lengths),
        "checked_max_samples": max(lengths),
        "checked_speakers": len(speakers),
    }

    print(
        f"{split_name:10s}",
        f"examples={len(dataset):,}",
        f"classes={len(path_labels)}",
        f"checked={checked}",
        f"length={min(lengths)}-{max(lengths)}",
        "PASS",
    )

assert len(all_labels) == EXPECTED_CLASSES, (
    f"expected {EXPECTED_CLASSES} classes, found {len(all_labels)}"
)

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

result = {
    "run": "speechcommands_v0.02_data_validation",
    "passed": True,
    "dataset": "speech_commands_v0.02",
    "raw_classes": len(all_labels),
    "labels": sorted(all_labels),
    "official_splits_disjoint": True,
    "sample_rate": EXPECTED_SAMPLE_RATE,
    "channels": 1,
    "splits": split_results,
    "torch_version": torch.__version__,
    "torchaudio_version": torchaudio.__version__,
    "next_protocol": (
        "Build a fixed 12-class or 35-class preprocessing pipeline, "
        "then run a small-batch Transformer overfit test."
    ),
}

RESULT_PATH.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("=" * 48)
print("SPEECH COMMANDS DATA VALIDATION: PASS")
print("saved:", RESULT_PATH)
