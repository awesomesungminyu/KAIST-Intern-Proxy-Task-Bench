import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


CONFIG = {
    "dataset": "Salesforce/wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "tokenizer": "gpt2",
    "context_length": 256,
    "num_blocks": 4,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "ffn_dim": 384,
    "learning_rate": 0.003,
    "max_steps": 1000,
    "pass_accuracy": 0.99,
    "required_streak": 5,
    "seed": 456,
}

RESULT_DIR = Path("/data/mqar/results")
RESULT_PATH = RESULT_DIR / "wikitext2_transformer_overfit.json"
CHECKPOINT_PATH = RESULT_DIR / "wikitext2_transformer_overfit.pt"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(scale + self.eps) * self.weight


def apply_rope(q, k):
    length = q.size(-2)
    head_dim = q.size(-1)

    positions = torch.arange(
        length,
        device=q.device,
        dtype=torch.float32,
    )

    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=q.device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )

    frequencies = torch.outer(positions, inv_freq)
    cosine = frequencies.cos()[None, None, :, :]
    sine = frequencies.sin()[None, None, :, :]

    def rotate(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated = torch.stack(
            (
                even * cosine - odd * sine,
                even * sine + odd * cosine,
            ),
            dim=-1,
        )

        return rotated.flatten(-2)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, length, dim = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch,
                length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        q, k = apply_rope(q, k)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, length, dim)
        return self.output(output)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.output(
            F.silu(self.gate(x)) * self.value(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.sequence_mixer = CausalSelfAttention(
            dim,
            num_heads,
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, x):
        x = x + self.sequence_mixer(
            self.attention_norm(x)
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        dim = CONFIG["d_model"]

        self.embedding = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    CONFIG["num_heads"],
                    CONFIG["ffn_dim"],
                )
                for _ in range(CONFIG["num_layers"])
            ]
        )

        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(
            dim,
            vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embedding.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, token_ids):
        x = self.embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)


torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading WikiText-2...")
dataset = load_dataset(
    CONFIG["dataset"],
    CONFIG["dataset_config"],
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["tokenizer"],
    use_fast=True,
    clean_up_tokenization_spaces=False,
)
tokenizer.model_max_length = 10**30

raw_text = "".join(dataset["train"]["text"])

token_ids = tokenizer(
    raw_text,
    add_special_tokens=False,
    return_attention_mask=False,
    verbose=False,
)["input_ids"]

token_ids.append(tokenizer.eos_token_id)

required_tokens = (
    CONFIG["num_blocks"]
    * CONFIG["context_length"]
    + 1
)

assert len(token_ids) >= required_tokens

tokens = torch.tensor(
    token_ids[:required_tokens],
    dtype=torch.long,
)

inputs = tokens[:-1].view(
    CONFIG["num_blocks"],
    CONFIG["context_length"],
)

labels = tokens[1:].view(
    CONFIG["num_blocks"],
    CONFIG["context_length"],
)

assert torch.equal(inputs[:, 1:], labels[:, :-1])

inputs = inputs.to(device)
labels = labels.to(device)

model = TransformerLM(len(tokenizer)).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=0.0,
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

print("device:", device)
print("input shape:", tuple(inputs.shape))
print("vocab size:", len(tokenizer))
print("parameters:", total_parameters)

start = time.perf_counter()
perfect_streak = 0
passed = False
final_loss = None
final_accuracy = None

for step in range(1, CONFIG["max_steps"] + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )
    optimizer.step()

    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        accuracy = (
            predictions == labels
        ).float().mean().item()

    final_loss = loss.item()
    final_accuracy = accuracy

    if accuracy >= CONFIG["pass_accuracy"]:
        perfect_streak += 1
    else:
        perfect_streak = 0

    if step == 1 or step % 25 == 0 or perfect_streak == 1:
        print(
            f"step={step:4d}",
            f"loss={final_loss:.6f}",
            f"accuracy={final_accuracy:.4f}",
            f"perfect_streak={perfect_streak}",
        )

    if perfect_streak >= CONFIG["required_streak"]:
        passed = True
        break

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed_seconds = time.perf_counter() - start

peak_vram_mib = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 2)
    if device.type == "cuda"
    else 0.0
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": CONFIG,
        "vocab_size": len(tokenizer),
    },
    CHECKPOINT_PATH,
)

summary = {
    "run": "wikitext2_transformer_overfit",
    "passed": passed,
    "config": CONFIG,
    "steps": step,
    "final_loss": final_loss,
    "final_accuracy": final_accuracy,
    "elapsed_seconds": elapsed_seconds,
    "device": str(device),
    "gpu": (
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else None
    ),
    "dtype": str(next(model.parameters()).dtype),
    "total_parameters": total_parameters,
    "peak_vram_mib": peak_vram_mib,
    "architecture": "Transformer++",
    "normalization": "RMSNorm",
    "ffn": "SwiGLU",
    "position_encoding": "RoPE inside attention",
    "attention": "causal softmax SDPA",
    "checkpoint": str(CHECKPOINT_PATH),
}

RESULT_PATH.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\n================================")
print(
    "WIKITEXT-2 TRANSFORMER OVERFIT:",
    "PASS" if passed else "FAIL",
)
print("steps:", step)
print("final loss:", final_loss)
print("final accuracy:", final_accuracy)
print("elapsed seconds:", round(elapsed_seconds, 3))
print("peak VRAM MiB:", round(peak_vram_mib, 2))
print("checkpoint:", CHECKPOINT_PATH)
print("result:", RESULT_PATH)
