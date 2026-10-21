from pathlib import Path

source = Path("overfit_parity.py")
output = Path("overfit_flipflop_transformer.py")
text = source.read_text(encoding="utf-8")

text = text.replace(
    "from validate_parity import make_dataset",
    "from validate_flipflop import make_dataset",
)

old_call = '''inputs, labels = make_dataset(
        num_examples=8,
        sequence_length=32,
        modulus=2,
        seed=123,
    )'''

new_call = '''inputs, labels, _, _ = make_dataset(
        num_examples=20,
        num_instructions=32,
        ignore_probability=0.80,
        seed=123,
    )'''

if old_call not in text:
    raise RuntimeError("기존 dataset 호출을 찾지 못했습니다.")

text = text.replace(old_call, new_call)
text = text.replace('"sequence_length": 32', '"sequence_length": 64')
text = text.replace(
    "parity_transformer_overfit",
    "flipflop_transformer_overfit",
)
text = text.replace('"examples": 8', '"examples": 20')
text = text.replace(
    '"sequence_length": 32,\n        "modulus": 2,',
    '"num_instructions": 32,\n        "ignore_probability": 0.80,',
)
text = text.replace(
    "PARITY TRANSFORMER OVERFIT:",
    "FLIP-FLOP TRANSFORMER OVERFIT:",
)

output.write_text(text, encoding="utf-8")
print("Created:", output.resolve())
