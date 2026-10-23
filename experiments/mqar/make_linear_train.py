from pathlib import Path


baseline_path = Path("/data/mqar/train_mqar_baseline.py")
overfit_path = Path("/data/mqar/overfit_linear_attention.py")
output_path = Path("/data/mqar/train_mqar_linear_attention.py")

baseline = baseline_path.read_text(encoding="utf-8")
linear_overfit = overfit_path.read_text(encoding="utf-8")


# Linear Attention 클래스 추출
linear_start = linear_overfit.index(
    "class CausalLinearAttention"
)
linear_end = linear_overfit.index(
    "class TransformerBlock",
    linear_start,
)
linear_mixer = linear_overfit[
    linear_start:linear_end
]


# Transformer attention 클래스 범위 확인
attention_start = baseline.index(
    "class CausalSelfAttention"
)
attention_end = baseline.index(
    "class TransformerBlock",
    attention_start,
)


# Attention mixer만 교체
candidate = (
    baseline[:attention_start]
    + linear_mixer
    + baseline[attention_end:]
)

candidate = candidate.replace(
    "CausalSelfAttention(",
    "CausalLinearAttention(",
)

candidate = candidate.replace(
    '"results/mqar_baseline"',
    '"results/mqar_linear_attention"',
)

candidate = candidate.replace(
    '"mqar_standard_transformer_baseline"',
    '"mqar_causal_linear_attention"',
)

candidate = candidate.replace(
    '"position_encoding": "RoPE inside attention",',
    '"position_encoding": "RoPE inside linear-attention mixer",',
)

candidate = candidate.replace(
    '"sequence_mixer_interface": "(B, T, D) -> (B, T, D)",',
    '''"sequence_mixer_interface": "(B, T, D) -> (B, T, D)",
    "sequence_mixer": "causal linear attention",
    "feature_map": "ELU+1",
    "theoretical_complexity": "O(T * d_head^2)",''',
)

output_path.write_text(
    candidate,
    encoding="utf-8",
)

print("Created:", output_path)
