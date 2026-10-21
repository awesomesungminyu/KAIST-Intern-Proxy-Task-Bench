"""RegBench 베이스라인: 작은 causal Transformer / LSTM.

Builder(vocab_size, max_len): tokens[B,T] -> logits[B,T,V], causal; num_params() -> int.

sm_75(Turing)에서는 FlashAttention-2가 없으므로 PyTorch 기본
scaled_dot_product_attention(SDPA, math/efficient 커널 자동 선택)만 사용.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.registry import ModelRegistry

MODELS = ModelRegistry("regbench")
register_model = MODELS.register


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, hd)
        # PyTorch 기본 SDPA: sm_75에서도 동작 (flash-attn 패키지 불필요)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """GPT-2 스타일, 기본값은 TVP 논문(2501.09240)의 RegBench 실험 설정
    (8 layers, 2 heads, d_model=128)을 따름."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 8,
        n_heads: int = 2,
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TinyLSTM(nn.Module):
    """HEXA-T 비교용 재귀형 베이스라인 (S4/RetNet 대신 가장 단순한 recurrent 기준선)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
        **_ignored,  # n_heads, max_len 등 Transformer 전용 인자를 무시하기 위함
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        x, _ = self.lstm(x)
        x = self.ln_f(x)
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@register_model("transformer")
def _transformer(vocab_size: int, max_len: int) -> nn.Module:
    return TinyGPT(vocab_size, max_len=max_len)


@register_model("lstm")
def _lstm(vocab_size: int, max_len: int) -> nn.Module:
    return TinyLSTM(vocab_size)


def build_model(name: str, vocab_size: int, max_len: int) -> nn.Module:
    return MODELS.build(name, vocab_size=vocab_size, max_len=max_len)
