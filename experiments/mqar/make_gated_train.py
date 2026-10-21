from pathlib import Path


baseline_path = Path(
    "/data/mqar/train_mqar_baseline.py"
)
gated_path = Path(
    "/data/mqar/overfit_gated_linear_attention.py"
)
output_path = Path(
    "/data/mqar/train_mqar_gated_linear_attention.py"
)

baseline = baseline_path.read_text(
    encoding="utf-8"
)
gated = gated_path.read_text(
    encoding="utf-8"
)

gated_start = gated.index(
    "class GatedLinearAttention"
)
gated_end = gated.index(
    "class TransformerBlock",
    gated_start,
)
gated_mixer = gated[gated_start:gated_end]

baseline_start = baseline.index(
    "class CausalSelfAttention"
)
baseline_end = baseline.index(
    "class TransformerBlock",
    baseline_start,
)

text = (
    baseline[:baseline_start]
    + gated_mixer
    + baseline[baseline_end:]
)

text = text.replace(
    "CausalSelfAttention("
    "CONFIG[\"d_model\"], CONFIG[\"num_heads\"]"
    ")",
    "GatedLinearAttention("
    "CONFIG[\"d_model\"], CONFIG[\"num_heads\"]"
    ")",
)

text = text.replace(
    "results/mqar_baseline",
    "results/mqar_gated_linear_attention",
)

text = text.replace(
    "mqar_standard_transformer_baseline",
    "mqar_gated_linear_attention",
)

text = text.replace(
    '"position_encoding": "RoPE inside attention",',
    '"position_encoding": "RoPE inside mixer",\n'
    '    "sequence_mixer": '
    '"gated recurrent linear attention",\n'
    '    "gate": "data-dependent per-head forget gate",\n'
    '    "theoretical_complexity": '
    '"O(T * d_head^2)",',
)

output_path.write_text(
    text,
    encoding="utf-8",
)

print("created:", output_path)