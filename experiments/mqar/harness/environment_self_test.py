
import json
import os
from pathlib import Path

seed = int(os.environ["HARNESS_SEED"])
output_dir = Path(os.environ["HARNESS_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

result = {
    "seed_received": seed,
    "output_directory_received": str(output_dir),
    "status": "ok",
}

path = output_dir / "self_test.json"
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
print(f"Saved: {path}")
