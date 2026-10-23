from pathlib import Path

SOURCE = Path("train_mqar_gated_linear_attention.py")


def convert(output_name, run_name, output_dir, signed):
    text = SOURCE.read_text(encoding="utf-8")

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
        '"mqar_gated_linear_attention"',
        f'"{run_name}"',
    )
    text = text.replace(
        "results/mqar_gated_linear_attention",
        output_dir,
    )

    if signed:
        old = '''gates = torch.sigmoid(
            self.forget_gate(x)
        )'''
        new = '''gates = (
            2.0 * torch.sigmoid(self.forget_gate(x)) - 1.0
        )'''

        if old not in text:
            raise RuntimeError("gate expression을 찾지 못했습니다.")

        text = text.replace(old, new)
        text = text.replace(
            '"gate": "data-dependent per-head forget gate"',
            '"gate": "signed data-dependent gate in [-1, 1]"',
        )
        text = text.replace(
            "BASELINE RUN:",
            "PARITY SIGNED GATED:",
        )
    else:
        text = text.replace(
            "BASELINE RUN:",
            "PARITY POSITIVE GATED:",
        )

    Path(output_name).write_text(text, encoding="utf-8")
    print("Created:", Path(output_name).resolve())


convert(
    output_name="train_parity_gated_positive.py",
    run_name="parity_gated_positive",
    output_dir="results/parity_gated_positive",
    signed=False,
)

convert(
    output_name="train_parity_gated_signed.py",
    run_name="parity_gated_signed",
    output_dir="results/parity_gated_signed",
    signed=True,
)
