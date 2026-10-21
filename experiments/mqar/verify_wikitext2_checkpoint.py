import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path("/data/mqar")
TRAIN_SCRIPT = ROOT / "train_wikitext2_transformer.py"
RESULT_DIR = ROOT / "results/wikitext2_transformer_baseline"
CHECKPOINT_PATH = RESULT_DIR / "best_model.pt"
SUMMARY_PATH = RESULT_DIR / "summary.json"
OUTPUT_PATH = RESULT_DIR / "checkpoint_validation.json"

source = TRAIN_SCRIPT.read_text(encoding="utf-8")
marker = 'set_seed(CONFIG["seed"])'
definitions = source[: source.index(marker)]

namespace = {"__name__": "wikitext2_model_definitions"}
exec(definitions, namespace)

CONFIG = namespace["CONFIG"]
TransformerLM = namespace["TransformerLM"]
TokenBlocks =isión = namespace["TokenBlocks"]
evaluate = namespace["evaluate"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)
summary = json.loads(
    SUMMARY_PATH.read_text(encoding="utf-8")
)

CONFIG.update(checkpoint["config"])

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["tokenizer"],
    use_fast=True,
    clean_up_tokenization_spaces=False,
)

valid_tokens = torch.load(
    ROOT / "data/wikitext2_gpt2/validation.pt"
)

valid_dataset = TokenBlocks(
    valid_tokens,
    CONFIG["context_length"],
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

model = TransformerLM(len(tokenizer)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

valid_loss, valid_ppl = evaluate(
    model,
    valid_loader,
    device,
)

saved_loss = float(summary["best_valid_loss"])
saved_ppl = float(summary["best_valid_ppl"])

loss_difference = abs(valid_loss - saved_loss)
ppl_difference = abs(valid_ppl - saved_ppl)

passed = (
    loss_difference < 1e-5
    and ppl_difference < 1e-3
)

result = {
    "run": "wikitext2_checkpoint_validation",
    "passed": passed,
    "checkpoint_epoch": checkpoint["epoch"],
    "reloaded_valid_loss": valid_loss,
    "saved_valid_loss": saved_loss,
    "loss_difference": loss_difference,
    "reloaded_valid_ppl": valid_ppl,
    "saved_valid_ppl": saved_ppl,
    "ppl_difference": ppl_difference,
    "device": str(device),
}

OUTPUT_PATH.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("checkpoint epoch:", checkpoint["epoch"])
print("reloaded valid loss:", valid_loss)
print("saved valid loss:", saved_loss)
print("reloaded valid PPL:", valid_ppl)
print("saved valid PPL:", saved_ppl)
print("================================")
print(
    "CHECKPOINT VALIDATION:",
    "PASS" if passed else "FAIL",
)
print("saved:", OUTPUT_PATH)

if not passed:
    raise SystemExit(1)
