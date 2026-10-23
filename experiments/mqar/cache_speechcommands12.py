import json
import runpy
import time
from pathlib import Path

import torch

ROOT = Path("/data/mqar")
CACHE_DIR = ROOT / "data" / "speechcommands12_cache"
RESULT_PATH = ROOT / "results" / "speechcommands12_cache_summary.json"

namespace = runpy.run_path(
    str(ROOT / "validate_speechcommands12.py")
)

SC12 = namespace["SC12"]
mel = namespace["mel"]
WANTED = namespace["WANTED"]

LABELS = sorted(WANTED) + ["unknown", "silence"]
LABEL_TO_ID = {
    label: index for index, label in enumerate(LABELS)
}

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

summary = {
    "run": "speechcommands12_feature_cache",
    "labels": LABELS,
    "feature": "normalized log-Mel",
    "dtype": "float16",
    "splits": {},
}

for split in ("training", "validation", "testing"):
    output_path = CACHE_DIR / f"{split}.pt"

    if output_path.exists():
        cached = torch.load(output_path, map_location="cpu")
        print(
            f"{split:10s}",
            f"already cached: {len(cached['targets']):,}",
            output_path,
        )
        summary["splits"][split] = {
            "examples": len(cached["targets"]),
            "path": str(output_path),
            "reused": True,
        }
        continue

    dataset = SC12(split)
    features = []
    targets = []
    start = time.perf_counter()

    print(f"\nCaching {split}: {len(dataset):,} examples")

    with torch.no_grad():
        for index in range(len(dataset)):
            waveform, sample_rate, label, _, _ = dataset[index]

            feature = torch.log(mel(waveform) + 1e-6)
            feature = (
                feature - feature.mean()
            ) / (feature.std() + 1e-6)

            features.append(feature.to(torch.float16))
            targets.append(LABEL_TO_ID[label])

            if (index + 1) % 2000 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"{split:10s}",
                    f"{index + 1:,}/{len(dataset):,}",
                    f"elapsed={elapsed / 60:.1f} min",
                )

    feature_tensor = torch.stack(features)
    target_tensor = torch.tensor(targets, dtype=torch.long)

    torch.save(
        {
            "features": feature_tensor,
            "targets": target_tensor,
            "labels": LABELS,
            "shape": list(feature_tensor.shape),
        },
        output_path,
    )

    elapsed = time.perf_counter() - start
    size_mib = output_path.stat().st_size / (1024 ** 2)

    summary["splits"][split] = {
        "examples": len(target_tensor),
        "shape": list(feature_tensor.shape),
        "elapsed_seconds": elapsed,
        "size_mib": size_mib,
        "path": str(output_path),
        "reused": False,
    }

    print(
        f"{split:10s}",
        "COMPLETE",
        f"shape={tuple(feature_tensor.shape)}",
        f"time={elapsed / 60:.1f} min",
        f"size={size_mib:.1f} MiB",
    )

RESULT_PATH.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("SPEECH COMMANDS FEATURE CACHE: COMPLETE")
print("cache:", CACHE_DIR)
print("summary:", RESULT_PATH)
