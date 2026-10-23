"""jelassi_copy 모델 등록 지점.

Builder(vocab_size): forward(tokens[B,T] int64) -> logits[B,T,V], causal.
평가 길이(최대 2*300+3 토큰)가 학습 context(220)보다 길다. 위치 정보는 모델 내부에서 처리하고
학습 길이에 묶인 학습형 위치 임베딩 표를 두지 않는다 (OOD 위치에서 index 오류가 난다).

transformer 는 Jelassi et al. 의 Hard-ALiBi 변형이다 (공식 synthetic_exps/models/hard_alibi_model.py):
위치 인코딩 없음(NoPE), 앞쪽 n_masked 개 헤드 중 i 번째 헤드는 직전 i+1 개 토큰만 본다.
나머지 헤드는 일반 causal attention. 공식 구현은 q·k 에 rotary 미적용 부분을 한 번 더 이어 붙이는
코드가 있으나(rotary_pct 차원 외 성분 중복) 여기서는 표준 q·k 를 쓴다.

mamba 는 mamba-ssm 커널 없이 순수 torch 순차 스캔이다 (sm_75 에 mamba-ssm 설치 불가).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.registry import ModelRegistry
from common_set import NAMES as COMMON_SET, TokenLM

MODELS = ModelRegistry("jelassi_copy")
register_model = MODELS.register


def hard_alibi_mask(n_heads, n_masked, T, device):
    """[H,T,T] additive mask. 헤드 i<n_masked 는 key j ∈ [q-i, q], 나머지는 j ≤ q."""
    q = torch.arange(T, device=device)[:, None]
    k = torch.arange(T, device=device)[None, :]
    window = torch.full((n_heads, 1, 1), T, device=device)
    window[:n_masked, 0, 0] = torch.arange(n_masked, device=device)
    allowed = (k <= q) & (q - k <= window)
    return torch.zeros(allowed.shape, device=device).masked_fill(~allowed, float("-inf"))


class HardAlibiAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_masked):
        super().__init__()
        self.n_heads, self.n_masked = n_heads, n_masked
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, C // self.n_heads).unbind(2)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_masked):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = HardAlibiAttention(d_model, n_heads, n_masked)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))


class HardAlibiTransformer(nn.Module):
    """pre-LN, GELU MLP. 공식 GPT-NeoX 는 parallel residual 이고 12층/16헤드/1024 폭이다."""

    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=8, n_masked=4):
        super().__init__()
        self.n_heads, self.n_masked = n_heads, n_masked
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, n_masked) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        mask = hard_alibi_mask(self.n_heads, self.n_masked, tokens.shape[1], tokens.device)
        h = self.emb(tokens)
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln_f(h))


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        return self.head(self.lstm(self.emb(tokens))[0])


class LinearScan(torch.autograd.Function):
    """H_t = a_t * H_{t-1} + b_t (H_{-1}=0), 시간축 dim=1. backward 는 같은 점화식을 역방향으로 푼다.

    step 마다 커널 하나만 띄우려고 autograd 대신 직접 썼다 (순차 autograd 루프보다 3~5배 빠름).
    순차 Python 루프 구현과 fp64 에서 forward·gradient 가 일치함을 테스트한다
    (tests/test_new_synthetic_tasks.py).
    """

    @staticmethod
    def forward(ctx, a, b):
        H = torch.empty_like(b)
        H[:, 0] = b[:, 0]
        for t in range(1, b.shape[1]):
            torch.addcmul(b[:, t], a[:, t], H[:, t - 1], out=H[:, t])
        ctx.save_for_backward(a, H)
        return H

    @staticmethod
    def backward(ctx, gH):
        a, H = ctx.saved_tensors
        G = gH.clone()
        for t in range(G.shape[1] - 2, -1, -1):
            G[:, t].addcmul_(a[:, t + 1], G[:, t + 1])
        ga = torch.zeros_like(a)
        ga[:, 1:] = G[:, 1:] * H[:, :-1]
        return ga, G


def selective_scan(u, delta, A, Bm, Cm):
    """h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t u_t,  y_t = C_t·h_t.  u,Δ:[B,L,E] A:[E,N] B,C:[B,L,N]."""
    H = LinearScan.apply(torch.exp(delta[..., None] * A), delta[..., None] * Bm[:, :, None, :] * u[..., None])
    return torch.einsum("blen,bln->ble", H, Cm)


class MambaMixer(nn.Module):
    """Mamba (Gu & Dao 2023) 블록의 참조 구현과 같은 구조·초기화. 커널만 순수 torch."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        e = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * e, bias=False)
        self.conv = nn.Conv1d(e, e, d_conv, groups=e, padding=d_conv - 1)
        self.x_proj = nn.Linear(e, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, e)
        dt = torch.exp(torch.rand(e) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # softplus 역함수
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(e, 1))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, d_model, bias=False)

    def forward(self, x):
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], -1)
        delta = F.softplus(self.dt_proj(dt))
        y = selective_scan(u, delta, -torch.exp(self.A_log), Bm, Cm) + u * self.D
        return self.out_proj(y * F.silu(z))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MambaLM(nn.Module):
    """mamba-ssm MambaLMHeadModel 과 같은 배치: pre-RMSNorm 잔차 Mamba 블록만 (MLP 없음)."""

    def __init__(self, vocab_size, d_model=256, n_layers=4, d_state=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.mixers = nn.ModuleList([MambaMixer(d_model, d_state) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        h = self.emb(tokens)
        for norm, mixer in zip(self.norms, self.mixers):
            h = h + mixer(norm(h))
        return self.head(self.norm_f(h))


@register_model("transformer")
def _transformer(vocab_size):
    return HardAlibiTransformer(vocab_size)


@register_model("lstm")
def _lstm(vocab_size):
    return LSTMLM(vocab_size)


@register_model("mamba")
def _mamba(vocab_size):
    return MambaLM(vocab_size)


# Longest evaluation sequence (2*300+3 = 603 tokens for the 201-300 bin) must fit hyena's filter length.
COMMON_SET_L_MAX = 1024


def _common_set(mixer):
    # Protocol v2 common architecture set: the Hard-ALiBi transformer's width/depth/heads, mixer-only
    # replacement. cs_transformer/cs_gqa use RoPE like every other task, not Hard-ALiBi.
    def build(vocab_size):
        return TokenLM(vocab_size, mixer, d_model=256, n_layers=4, n_heads=8, l_max=COMMON_SET_L_MAX)
    return build


for _mixer in COMMON_SET:
    register_model("cs_" + _mixer)(_common_set(_mixer))
