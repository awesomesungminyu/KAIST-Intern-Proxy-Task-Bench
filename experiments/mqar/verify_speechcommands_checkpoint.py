import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/data/mqar")
TRAIN_SCRIPT = ROOT / "train_speechcommands_transformer.py"
OUTPUT_DIR = ROOT / "results" / "speechcommands_transformer_baseline"

source = TRAIN_SCRIPT.read_text(encoding="utf-8")
marker = 'model = KeywordTransformer().to(device)'
definitions = source[:source.index(marker)]

namespace = {"__name__": "speechcommands_model_definitions"}
exec(compile(definitions, str(TRAIN_SCRIPT), "exec"), namespace)

KeywordTransformer = namespace["KeywordTransformer"]
valid_loader = namespace["valid_loader"]
device = namespace["device"]

checkpoint = torch.load(
    OUTPUT_DIR / "best_model.pt",
    map_location=device,
)
saved_summary = json.loads(
    (OUTPUT_DIR / "summary.json").read_text(encoding="utf-8")
)

model = KeywordTransformer().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

loss_sum = 0.0
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
        loss_sum += F.cross_entropy(
            logits,
            targets,
            reduction="sum",
        ).item()
        correct += (
            logits.argmax(dim=-1) == targets
        ).sum().item()
        total += len(targets)

reloaded_loss = loss_sum / total
reloaded_accuracy = correct / total
saved_loss = saved_summary["best_valid_loss"]
saved_accuracy = saved_summary["best_valid_accuracy"]

passed = (
    abs(reloaded_loss - saved_loss) < 1e-6
    and abs(reloaded_accuracy - saved_accuracy) < 1e-12
)

result = {
    "run": "speechcommands_checkpoint_validation",
    "passed": passed,
    "checkpoint_epoch": checkpoint["epoch"],
    "reloaded_valid_loss": reloaded_loss,
    "saved_valid_loss": saved_loss,
    "reloaded_valid_accuracy": reloaded_accuracy,
    "saved_valid_accuracy": saved_accuracy,
}

result_path = OUTPUT_DIR / "checkpoint_validation.json"
result_path.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("checkpoint epoch:", checkpoint["epoch"])
print("reloaded valid loss:", reloaded_loss)
print("saved valid loss:", saved_loss)
print("reloaded valid accuracy:", reloaded_accuracy)
print("saved valid accuracy:", saved_accuracy)
print("=" * 52)
print("CHECKPOINT VALIDATION:", "PASS" if passed else "FAIL")
print("saved:", result_path)
