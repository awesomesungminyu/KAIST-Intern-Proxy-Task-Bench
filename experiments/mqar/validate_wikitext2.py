import json
import time
from pathlib import Path

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TOKENIZER_NAME = "gpt2"
CONTEXT_LENGTH = 256
OUTPUT_PATH = Path(
    "/data/mqar/results/wikitext2_data_validation.json"
)


def encode_split(dataset, split, tokenizer):
    texts = dataset[split]["text"]
    nonempty_rows = sum(bool(text.strip()) for text in texts)

    # 원본 줄바꿈과 빈 줄을 유지한 split별 단일 token stream
    raw_text = "".join(texts)
    token_ids = tokenizer(
        raw_text,
        add_special_tokens=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
    token_ids.append(tokenizer.eos_token_id)

    usable_tokens = ((len(token_ids) - 1) // CONTEXT_LENGTH) * CONTEXT_LENGTH

    inputs = torch.tensor(
        token_ids[:usable_tokens],
        dtype=torch.long,
    ).view(-1, CONTEXT_LENGTH)

    labels = torch.tensor(
        token_ids[1 : usable_tokens + 1],
        dtype=torch.long,
    ).view(-1, CONTEXT_LENGTH)

    return {
        "rows": len(texts),
        "nonempty_rows": nonempty_rows,
        "tokens": len(token_ids),
        "usable_tokens": usable_tokens,
        "blocks": len(inputs),
        "inputs": inputs,
        "labels": labels,
    }

start = time.perf_counter()

print("Loading WikiText-2...")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

print("Loading GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True,
    clean_up_tokenization_spaces=False,
)
tokenizer.model_max_length = 10**30

assert tokenizer.eos_token_id is not None
assert set(dataset.keys()) == {"train", "validation", "test"}

results = {}

for split in ("train", "validation", "test"):
    data = encode_split(dataset, split, tokenizer)

    inputs = data.pop("inputs")
    labels = data.pop("labels")

    assert inputs.shape == labels.shape
    assert inputs.shape[1] == CONTEXT_LENGTH
    assert torch.equal(inputs[:, 1:], labels[:, :-1])
    assert inputs.min().item() >= 0
    assert inputs.max().item() < len(tokenizer)

    data["fingerprint"] = dataset[split]._fingerprint
    data["shift_check"] = "PASS"
    results[split] = data

    print(
        f"{split:10s}",
        f"rows={data['rows']:,}",
        f"nonempty={data['nonempty_rows']:,}",
        f"tokens={data['tokens']:,}",
        f"blocks={data['blocks']:,}",
        "shift=PASS",
    )

sample_text = dataset["train"][1]["text"]
sample_a = tokenizer.encode(sample_text, add_special_tokens=False)
sample_b = tokenizer.encode(sample_text, add_special_tokens=False)
assert sample_a == sample_b

summary = {
    "run": "wikitext2_data_validation",
    "passed": True,
    "dataset": DATASET_NAME,
    "dataset_config": DATASET_CONFIG,
    "tokenizer": TOKENIZER_NAME,
    "tokenizer_class": tokenizer.__class__.__name__,
    "vocab_size": len(tokenizer),
    "eos_token_id": tokenizer.eos_token_id,
    "context_length": CONTEXT_LENGTH,
    "document_boundary": "original WikiText raw newlines preserved",
    "boundary_policy": "one continuous token stream per split; EOS only at split end",
    "splits": results,
    "deterministic_tokenization": True,
    "datasets_version": datasets.__version__,
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__,
    "elapsed_seconds": time.perf_counter() - start,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n================================")
print("WIKITEXT-2 DATA VALIDATION: PASS")
print("saved:", OUTPUT_PATH)