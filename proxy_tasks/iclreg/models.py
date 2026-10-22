"""iclreg 모델 등록 지점.

다른 태스크와 달리 입력이 토큰이 아니라 연속 벡터다.
Builder(n_dims, n_points): forward(xs[B,K,d] float32, ys[B,K] float32) -> preds[B,K] float32.
preds[:, i] 는 (x_1, y_1, ..., x_i, y_i, x_{i+1}) 만 보고 낸 y_{i+1} 예측이다 (0-index 로 ys[:, i]).
즉 모델은 x 와 y 를 [x_1, y_1, x_2, y_2, ...] 로 번갈아 놓은 길이 2K 시퀀스를 causal 로 처리하고
x 위치의 출력을 읽는다. ys[:, i] 가 preds[:, i] 에 새면 정답 누출이다.

x 는 nn.Linear(d, D), 스칼라 y 는 별도 nn.Linear(1, D) 로 같은 폭 D 에 올린다 (Interleave).
Garg et al. 은 y 를 [y, 0, ..., 0] 으로 채워 x 와 같은 read-in 을 쓴다. 별도 y 임베딩은
그 일반화다 (y 축 가중치와 bias 를 따로 학습).

sm_75 에서는 PyTorch SDPA 만 쓴다. mamba 는 순수 torch 스캔이 45000 step 예산에서 seed 당
30분을 넘어(약 112 ms/step) 등록하지 않았다. jelassi_copy/models.py 의 구현을 옮겨 쓸 수 있다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.registry import ModelRegistry

MODELS = ModelRegistry("iclreg")
register_model = MODELS.register


class Interleave(nn.Module):
    """xs[B,K,d], ys[B,K] -> z[B,2K,D] = [x_1, y_1, x_2, y_2, ...]."""

    def __init__(self, n_dims, width):
        super().__init__()
        self.x_in = nn.Linear(n_dims, width)
        self.y_in = nn.Linear(1, width)

    def forward(self, xs, ys):
        B, K, _ = xs.shape
        z = torch.stack([self.x_in(xs), self.y_in(ys[..., None])], dim=2)
        return z.reshape(B, 2 * K, -1)


def read_x_positions(h, head):
    """x 위치(짝수 인덱스)의 출력만 읽는다. [B,2K,D] -> [B,K]."""
    return head(h[:, ::2])[..., 0]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, C // self.n_heads).unbind(2)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TransformerRegressor(nn.Module):
    """GPT-2 형 (pre-LN, GELU, 학습형 위치 임베딩). Garg et al. 은 12층/8헤드/256 폭."""

    def __init__(self, n_dims, n_points, d_model=256, n_layers=6, n_heads=8):
        super().__init__()
        self.read_in = Interleave(n_dims, d_model)
        self.pos = nn.Embedding(2 * n_points, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, xs, ys):
        z = self.read_in(xs, ys)
        h = z + self.pos(torch.arange(z.shape[1], device=z.device))
        for block in self.blocks:
            h = block(h)
        return read_x_positions(self.ln_f(h), self.head)


class LSTMRegressor(nn.Module):
    def __init__(self, n_dims, n_points, d_model=512, n_layers=3):
        super().__init__()
        self.read_in = Interleave(n_dims, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.head = nn.Linear(d_model, 1)

    def forward(self, xs, ys):
        return read_x_positions(self.lstm(self.read_in(xs, ys))[0], self.head)


@register_model("transformer")
def _transformer(n_dims, n_points):
    return TransformerRegressor(n_dims, n_points)


@register_model("lstm")
def _lstm(n_dims, n_points):
    return LSTMRegressor(n_dims, n_points)

