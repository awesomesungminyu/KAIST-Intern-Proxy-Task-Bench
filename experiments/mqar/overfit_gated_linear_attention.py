import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from zoology.data.multiquery_ar import MQARConfig


# ============================================================
# Configuration
# ============================================================

DATA_SEED = 123
MODEL_SEED = 456

VOCAB_SIZE = 64
SEQ_LEN = 32
NUM_KV_PAIRS = 4
NUM_EXAMPLES = 20

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 512

LEARNING_RATE = 3e-3
MAX_STEPS = 3000
REQUIRED_PERFECT_STEPS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
DTYPE = torch.float32


# ============================================================
# Reproducibility
# ============================================================

random.seed(MODEL_SEED)
np.random.seed(MODEL_SEED)
torch.manual_seed(MODEL_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MODEL_SEED)
    torch.cuda.reset_peak_memory_stats()


# ============================================================
# Replaceable sequence mixer interface
# ============================================================

class SequenceMixer(nn.Module):
    def forward(self, x):
        """
        Input:  (batch, sequence, dimension)
        Output: (batch, sequence, dimension)
        """
        raise NotImplementedError


# ============================================================
# RoPE
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10_000):
        super().__init__()

        assert head_dim % 2 == 0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(self, q, k):
        sequence_length = q.size(2)

        positions = torch.arange(
            sequence_length,
            device=q.device,
            dtype=torch.float32,
        )

        angles = torch.outer(
            positions,
            self.inv_freq,
        )

        cos = angles.cos().to(q.dtype)[None, None]
        sin = angles.sin().to(q.dtype)[None, None]

        def rotate(x):
            even = x[..., 0::2]
            odd = x[..., 1::2]

            rotated = torch.stack(
                (
                    even * cos - odd * sin,
                    even * sin + odd * cos,
                ),
                dim=-1,
            )

            return rotated.flatten(-2)

        return rotate(q), rotate(k)


# ============================================================
# Causal Linear Attention
# ============================================================
class GatedLinearAttention(SequenceMixer):
    """
    Pure PyTorch gated recurrent linear attention.

    A learned forget gate controls how much previous KV state
    remains at each position.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        eps=1e-6,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
            bias=False,
        )
        self.output = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # One data-dependent retention gate per head.
        self.forget_gate = nn.Linear(
            d_model,
            num_heads,
            bias=True,
        )

        # Start with a gate close to 1 so long-range
        # information is retained early in training.
        nn.init.zeros_(self.forget_gate.weight)
        nn.init.constant_(self.forget_gate.bias, 3.0)

        self.rope = RotaryEmbedding(self.head_dim)

    @staticmethod
    def feature_map(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        q, k = self.rope(q, k)
        q = self.feature_map(q)
        k = self.feature_map(k)

        gates = torch.sigmoid(
            self.forget_gate(x)
        )
        gates = gates.transpose(1, 2).unsqueeze(-1)

        kv_state = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        k_state = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []

        for position in range(sequence_length):
            gate = gates[:, :, position]

            key = k[:, :, position]
            value = v[:, :, position]
            query = q[:, :, position]

            kv_update = torch.einsum(
                "bhd,bhe->bhde",
                key,
                value,
            )

            kv_state = (
                gate.unsqueeze(-1) * kv_state
                + kv_update
            )
            k_state = gate * k_state + key

            numerator = torch.einsum(
                "bhd,bhde->bhe",
                query,
                kv_state,
            )

            denominator = torch.einsum(
                "bhd,bhd->bh",
                query,
                k_state,
            ).unsqueeze(-1)

            output = numerator / denominator.clamp_min(
                self.eps
            )
            outputs.append(output)

        mixed = torch.stack(outputs, dim=2)
        mixed = (
            mixed.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                d_model,
            )
        )

        return self.output(mixed)
# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.mixer_norm = nn.LayerNorm(D_MODEL)
        self.sequence_mixer = GatedLinearAttention(
            D_MODEL,
            NUM_HEADS,
        )

        self.ffn_norm = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, D_MODEL),
        )

    def forward(self, x):
        x = x + self.sequence_mixer(
            self.mixer_norm(x)
        )
        x = x + self.ffn(
            self.ffn_norm(x)
        )
        return x


# ============================================================
# Language model
# ============================================================

class LinearAttentionLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL,
        )

        self.blocks = nn.ModuleList(
            TransformerBlock()
            for _ in range(NUM_LAYERS)
        )

        self.final_norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(
            D_MODEL,
            VOCAB_SIZE,
            bias=False,
        )

        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(
            module,
            (nn.Linear, nn.Embedding),
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        hidden = self.token_embedding(token_ids)

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)


# ============================================================
# MQAR data
# ============================================================

data_config = MQARConfig(
    vocab_size=VOCAB_SIZE,
    num_examples=NUM_EXAMPLES,
    input_seq_len=SEQ_LEN,
    num_kv_pairs=NUM_KV_PAIRS,
    random_non_queries=False,
    include_slices=True,
)

dataset = data_config.build(seed=DATA_SEED)

inputs = dataset.inputs.to(DEVICE)
labels = dataset.labels.to(DEVICE)

assert inputs.shape == (
    NUM_EXAMPLES,
    SEQ_LEN,
)
assert labels.shape == (
    NUM_EXAMPLES,
    SEQ_LEN,
)

target_count = int(
    (labels != -100).sum()
)

assert target_count == (
    NUM_EXAMPLES * NUM_KV_PAIRS
)


# ============================================================
# Model
# ============================================================

model = LinearAttentionLM().to(
    device=DEVICE,
    dtype=DTYPE,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0,
)

parameter_groups = {
    "token_embedding": sum(
        p.numel()
        for p in model.token_embedding.parameters()
    ),
    "sequence_mixers": sum(
        p.numel()
        for block in model.blocks
        for p in block.sequence_mixer.parameters()
    ),
    "ffn": sum(
        p.numel()
        for block in model.blocks
        for p in block.ffn.parameters()
    ),
    "normalization": sum(
        p.numel()
        for name, p in model.named_parameters()
        if "norm" in name
    ),
    "lm_head": sum(
        p.numel()
        for p in model.lm_head.parameters()
    ),
}

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

assert sum(parameter_groups.values()) == total_parameters

print("device:", DEVICE)
print("dtype:", DTYPE)
print("input shape:", tuple(inputs.shape))
print("supervised targets:", target_count)
print("parameter groups:", parameter_groups)
print("total parameters:", total_parameters)


# ============================================================
# Accuracy
# ============================================================

def target_accuracy(logits, targets):
    predictions = logits.argmax(dim=-1)
    mask = targets != -100

    correct = (
        predictions[mask] == targets[mask]
    ).sum()

    return correct.float() / mask.sum()


# ============================================================
# Fixed-batch overfit test
# ============================================================

model.train()

perfect_streak = 0
passed = False
final_loss = None
final_accuracy = None

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

start_time = time.perf_counter()

for step in range(1, MAX_STEPS + 1):
    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        labels.reshape(-1),
        ignore_index=-100,
    )

    accuracy = target_accuracy(
        logits,
        labels,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    final_loss = float(loss.detach())
    final_accuracy = float(accuracy.detach())

    if final_accuracy == 1.0:
        perfect_streak += 1
    else:
        perfect_streak = 0

    if (
        step == 1
        or step % 100 == 0
        or perfect_streak == 1
    ):
        print(
            f"step={step:4d} "
            f"loss={final_loss:.6f} "
            f"accuracy={final_accuracy:.4f} "
            f"perfect_streak={perfect_streak}"
        )

    if perfect_streak >= REQUIRED_PERFECT_STEPS:
        passed = True
        break

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

elapsed_seconds = (
    time.perf_counter() - start_time
)

steps_completed = step

peak_vram_bytes = (
    torch.cuda.max_memory_allocated()
    if DEVICE.type == "cuda"
    else 0
)


# ============================================================
# Zoology version
# ============================================================

try:
    zoology_commit = subprocess.check_output(
        [
            "git",
            "-C",
            "/data/mqar/zoology-src",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
except Exception:
    zoology_commit = None


# ============================================================
# Save results
# ============================================================

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

checkpoint_path = (
    output_dir / "gated_linear_attention_overfit.pt"
)
result_path = (
    output_dir / "gated_linear_attention_overfit.json"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "data_config": data_config.model_dump(),
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
        "model_config": {
            "vocab_size": VOCAB_SIZE,
            "sequence_length": SEQ_LEN,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ffn_dim": FFN_DIM,
            "sequence_mixer": (
                "causal linear attention"
            ),
            "feature_map": "ELU+1",
            "position_encoding": (
                "RoPE inside mixer"
            ),
            "normalization": "Pre-LN",
        },
    },
    checkpoint_path,
)

result = {
    "test": (
        "fixed_20_example_linear_attention_overfit"
    ),
    "passed": passed,
    "required_perfect_steps": (
        REQUIRED_PERFECT_STEPS
    ),
    "steps_completed": steps_completed,
    "final_loss": final_loss,
    "final_accuracy": final_accuracy,
    "perfect_streak": perfect_streak,
    "elapsed_seconds": elapsed_seconds,
    "steps_per_second": (
        steps_completed / elapsed_seconds
    ),
    "device": str(DEVICE),
    "gpu": (
        torch.cuda.get_device_name(0)
        if DEVICE.type == "cuda"
        else None
    ),
    "compute_capability": (
        list(torch.cuda.get_device_capability(0))
        if DEVICE.type == "cuda"
        else None
    ),
    "dtype": str(DTYPE),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "peak_vram_bytes": peak_vram_bytes,
    "peak_vram_mib": (
        peak_vram_bytes / (1024 ** 2)
    ),
    "total_parameters": total_parameters,
    "parameter_groups": parameter_groups,
    "data_seed": DATA_SEED,
    "model_seed": MODEL_SEED,
    "data_config": data_config.model_dump(),
    "sequence_mixer": (
        "causal linear attention"
    ),
    "feature_map": "ELU+1",
    "theoretical_complexity": "O(T * d_head^2)",
    "zoology_commit": zoology_commit,
    "checkpoint": str(
        checkpoint_path.resolve()
    ),
}

result_path.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\n==============================")
print(
    "GATED LINEAR ATTENTION OVERFIT:",
    "PASS" if passed else "FAIL",
)
print("steps:", steps_completed)
print("final loss:", final_loss)
print("final accuracy:", final_accuracy)
print(
    "elapsed seconds:",
    round(elapsed_seconds, 3),
)
print(
    "peak VRAM MiB:",
    round(
        peak_vram_bytes / (1024 ** 2),
        2,
    ),
)
print("checkpoint:", checkpoint_path.resolve())
print("result:", result_path.resolve())

if not passed:
    raise SystemExit(1)