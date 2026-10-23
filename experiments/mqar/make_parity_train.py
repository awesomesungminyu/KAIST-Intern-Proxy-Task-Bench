from pathlib import Path

source = Path("train_mqar_baseline.py")
output = Path("train_parity_baseline.py")
text = source.read_text(encoding="utf-8")

text = text.replace(
    "from zoology.data.multiquery_ar import MQARConfig",
    "from validate_parity import make_dataset as make_parity_dataset",
)

start = text.index("def make_dataset(")
end = text.index("\n\n@torch.no_grad()", start)

replacement = '''def make_dataset(num_examples, seed):
    return TensorDataset(
        *make_parity_dataset(
            num_examples=num_examples,
            sequence_length=CONFIG["sequence_length"],
            modulus=CONFIG["modulus"],
            seed=seed,
        )
    )
'''

text = text[:start] + replacement + text[end:]

text = text.replace('"vocab_size": 8192', '"vocab_size": 12')
text = text.replace('"sequence_length": 64', '"sequence_length": 32')
text = text.replace('"num_kv_pairs": 8', '"modulus": 2')
text = text.replace(
    '"mqar_standard_transformer_baseline"',
    '"parity_standard_transformer_baseline"',
)
text = text.replace(
    'results/mqar_baseline',
    'results/parity_baseline',
)
text = text.replace(
    'BASELINE RUN:',
    'PARITY BASELINE:',
)

output.write_text(text, encoding="utf-8")
print("Created:", output.resolve())
