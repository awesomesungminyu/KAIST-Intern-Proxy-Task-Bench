import json
from pathlib import Path

import torch
from zoology.data.multiquery_ar import MQARConfig

SEED = 123
NUM_EXAMPLES = 5
VOCAB_SIZE = 32
SEQ_LEN = 16
NUM_KV_PAIRS = 2

config = MQARConfig(
    vocab_size=VOCAB_SIZE,
    num_examples=NUM_EXAMPLES,
    input_seq_len=SEQ_LEN,
    num_kv_pairs=NUM_KV_PAIRS,
    random_non_queries=False,
    include_slices=True,
)

data = config.build(seed=SEED)

assert data.inputs.shape == (NUM_EXAMPLES, SEQ_LEN)
assert data.labels.shape == (NUM_EXAMPLES, SEQ_LEN)
assert data.inputs.dtype == torch.int64
assert data.labels.dtype == torch.int64

records = []
all_passed = True
context_length = NUM_KV_PAIRS * 2

for sample_idx, (inputs, labels) in enumerate(
    zip(data.inputs, data.labels)
):
    kv_map = {
        int(inputs[pos]): int(inputs[pos + 1])
        for pos in range(0, context_length, 2)
    }

    target_positions = torch.where(labels != -100)[0].tolist()
    checks = []

    for position in target_positions:
        query = int(inputs[position])
        answer = int(labels[position])
        expected = kv_map.get(query)

        checks.append({
            "position": position,
            "query": query,
            "answer": answer,
            "expected": expected,
            "passed": expected == answer,
        })

    sample_passed = (
        len(target_positions) == NUM_KV_PAIRS
        and all(check["passed"] for check in checks)
        and torch.all(labels[:context_length] == -100).item()
    )
    all_passed &= sample_passed

    record = {
        "sample": sample_idx,
        "inputs": inputs.tolist(),
        "labels": labels.tolist(),
        "kv_map": kv_map,
        "target_positions": target_positions,
        "checks": checks,
        "passed": sample_passed,
    }
    records.append(record)

    print(f"\n=== SAMPLE {sample_idx} ===")
    print("inputs :", inputs.tolist())
    print("labels :", labels.tolist())
    print("KV map :", kv_map)

    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(
            f"position={check['position']:2d} "
            f"query={check['query']:2d} "
            f"answer={check['answer']:2d} "
            f"expected={check['expected']} [{status}]"
        )

    print("sample result:", "PASS" if sample_passed else "FAIL")

result = {
    "generator": "zoology.data.multiquery_ar.MQARConfig",
    "seed": SEED,
    "config": config.model_dump(),
    "input_shape": list(data.inputs.shape),
    "label_shape": list(data.labels.shape),
    "causal_shift_interpretation": (
        "The label aligned with each query token is the next-token value "
        "that the causal model must predict."
    ),
    "all_passed": bool(all_passed),
    "samples": records,
}

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "mqar_data_validation.json"
output_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n==============================")
print("OVERALL:", "PASS" if all_passed else "FAIL")
print("saved:", output_path.resolve())

if not all_passed:
    raise SystemExit(1)
