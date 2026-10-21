from pathlib import Path

source = Path("train_parity_baseline.py")
output = Path("train_flipflop_transformer.py")
text = source.read_text(encoding="utf-8")

text = text.replace(
    "from validate_parity import make_dataset as make_parity_dataset",
    "from validate_flipflop import make_dataset as make_flipflop_dataset",
)

start = text.index("def make_dataset(")
end = text.index("\n\n@torch.no_grad()", start)

replacement = '''def make_dataset(num_examples, seed):
    inputs, labels, _, _ = make_flipflop_dataset(
        num_examples=num_examples,
        num_instructions=CONFIG["num_instructions"],
        ignore_probability=CONFIG["ignore_probability"],
        seed=seed,
    )
    return TensorDataset(inputs, labels)
'''

text = text[:start] + replacement + text[end:]

text = text.replace('"sequence_length": 32', '"sequence_length": 128')
text = text.replace(
    '"modulus": 2',
    '"num_instructions": 64,\n    "ignore_probability": 0.80',
)
text = text.replace(
    "parity_standard_transformer_baseline",
    "flipflop_standard_transformer",
)
text = text.replace(
    "results/parity_baseline",
    "results/flipflop_transformer",
)
text = text.replace(
    "PARITY BASELINE:",
    "FLIP-FLOP TRANSFORMER:",
)

output.write_text(text, encoding="utf-8")
print("Created:", output.resolve())
