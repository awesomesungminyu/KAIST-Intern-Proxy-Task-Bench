"""Causal sequence mixers: float [B,T,d] -> float [B,T,d].

Attention uses RoPE. Mamba uses a pure-PyTorch scan; LSTM uses torch.nn.LSTM.
Hyena and S4D source licences are preserved in licenses/. RetNet uses retention.py.
Internal widths are chosen by backbone.mixer_width to match the Transformer block budget.
"""
import math

import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F

from .hyena import HyenaOperator
from .retention import retention
from .s4d import S4D

NAMES = ["transformer", "gqa", "linear_attention", "gla", "mamba", "hyena", "lstm", "s4d", "retnet"]
# Official internal width of each matchable mixer, as a function of d_model.
OFFICIAL_WIDTH = {
    "mamba": lambda d: 2 * d,      # expand = 2
    "lstm": lambda d: d,
    "retnet": lambda d: 2 * d,     # fla expand_v = 2
    "hyena": lambda d: 64,         # safari filter_order
    "s4d": lambda d: 64,           # s4d.py d_state
}


# ----------------------------------------------------------------------------
# Attention family
# ----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10_000):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k):
        length = q.size(2)
        positions = torch.arange(length, device=q.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos().to(q.dtype)[None, None]
        sin = angles.sin().to(q.dtype)[None, None]

        def rotate(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = self.rope(q, k)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.output(output.transpose(1, 2).contiguous().view(batch, length, width))


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert d_model % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.output(output.transpose(1, 2).contiguous().view(batch, length, width))


class CausalLinearAttention(nn.Module):
    """phi(x) = ELU(x) + 1 with cumulative K and KV prefix states."""

    def __init__(self, d_model, num_heads, eps=1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = self.rope(q, k)
        q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
        # Bound the B*H*T*d*d prefix tensor; carry the exact prefix state
        # across time tiles. Checkpoint recomputes tile intermediates in backward.
        ks = k.new_zeros(batch, self.num_heads, self.head_dim)
        kvs = k.new_zeros(batch, self.num_heads, self.head_dim, self.head_dim)
        pieces = []
        def tile(qt, kt, vt, previous_k, previous_kv):
            kv = torch.einsum("bhtd,bhte->bhtde", kt, vt).cumsum(2) + previous_kv.unsqueeze(2)
            kp = kt.cumsum(2) + previous_k.unsqueeze(2)
            num = torch.einsum("bhtd,bhtde->bhte", qt, kv)
            den = torch.einsum("bhtd,bhtd->bht", qt, kp).unsqueeze(-1)
            return num / den.clamp_min(self.eps), kp[:, :, -1].clone(), kv[:, :, -1].clone()
        for start in range(0, length, 64):
            args = (q[:, :, start:start+64], k[:, :, start:start+64], v[:, :, start:start+64], ks, kvs)
            part, ks, kvs = checkpoint(tile, *args, use_reentrant=False) if torch.is_grad_enabled() else tile(*args)
            pieces.append(part)
        mixed = torch.cat(pieces, dim=2)
        return self.output(mixed.transpose(1, 2).contiguous().view(batch, length, width))


# ----------------------------------------------------------------------------
# GLA (from proxy_tasks/tier0/models.py LinearAttnDecay, gate_mode="gla")
# ----------------------------------------------------------------------------

class GatedLinearAttention(nn.Module):
    """Head-wise input-dependent decay, D[i,j] = exp(cumlogg_i - cumlogg_j) (j<=i), O(L^2) vectorised."""

    def __init__(self, d, n_head):
        super().__init__()
        self.h, self.dh = n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.g = nn.Linear(d, n_head, bias=True)
        nn.init.constant_(self.g.bias, 2.0)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        q, k = F.elu(q) + 1, F.elu(k) + 1
        logg = -F.softplus(self.g(x)).transpose(1, 2)
        c = logg.cumsum(-1)
        pieces = []
        # Each query tile still attends to ALL keys. Preserve the original
        # clamp(-30,0), causal mask and denominator epsilon, including its tail.
        def tile(qt, all_k, all_v, ct, all_c, causal):
            dec = (ct[..., :, None] - all_c[..., None, :]).clamp(min=-30, max=0).exp()
            a = (qt @ all_k.transpose(-1, -2)) * dec.masked_fill(~causal, 0.0)
            a = a / (a.sum(-1, keepdim=True) + 1e-6)
            return a @ all_v
        keys = torch.arange(L, device=x.device)
        for start in range(0, L, 64):
            stop = min(start+64, L)
            causal = keys[None, :] <= keys[start:stop, None]
            args = (q[:, :, start:stop], k, v, c[:, :, start:stop], c, causal)
            pieces.append(checkpoint(tile, *args, use_reentrant=False) if torch.is_grad_enabled() else tile(*args))
        return self.o(torch.cat(pieces, dim=2).transpose(1, 2).reshape(B, L, D))



# ----------------------------------------------------------------------------
# Mamba (from proxy_tasks/T1_R0_02/models.py)
# ----------------------------------------------------------------------------

class LinearScan(torch.autograd.Function):
    """H_t = a_t * H_{t-1} + b_t (H_{-1}=0) along dim=1; backward solves the same recurrence in reverse."""

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
    """h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t,  y_t = C_t . h_t.  u,delta:[B,L,E] A:[E,N] B,C:[B,L,N]."""
    H = LinearScan.apply(torch.exp(delta[..., None] * A), delta[..., None] * Bm[:, :, None, :] * u[..., None])
    return torch.einsum("blen,bln->ble", H, Cm)


class MambaMixer(nn.Module):
    """Mamba (Gu & Dao 2023) block structure and initialisation of the reference implementation; pure-torch scan."""

    def __init__(self, d_model, d_inner, d_state=16, d_conv=4):
        super().__init__()
        e = d_inner
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * e, bias=False)
        self.conv = nn.Conv1d(e, e, d_conv, groups=e, padding=d_conv - 1)
        self.x_proj = nn.Linear(e, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, e)
        dt = torch.exp(torch.rand(e) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
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


# ----------------------------------------------------------------------------
# RetNet (fla-org/flash-linear-attention v0.3.1 fla/layers/multiscale_retention.py, MIT)
# ----------------------------------------------------------------------------

class RetNetMixer(nn.Module):
    """MultiScaleRetention with fla defaults: expand_k=1, no short conv, swish output gate,
    gated RMSNorm (eps 1e-5) per value head, non-interleaved rotary on q/k, fixed decay 1 - 2^(-5-h).
    retention_backend: see retention.retention ("auto" = fla on CUDA, "torch" = parallel form everywhere)."""

    def __init__(self, d_model, num_heads, value_dim, base=10_000.0, retention_backend="auto"):
        super().__init__()
        self.retention_backend = retention_backend
        assert d_model % num_heads == 0 and value_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_k_dim = d_model // num_heads
        self.head_v_dim = value_dim // num_heads
        assert self.head_k_dim % 2 == 0 and self.head_k_dim <= 256
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, value_dim, bias=False)
        self.o_proj = nn.Linear(value_dim, d_model, bias=False)
        self.g_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.norm_eps = 1e-5
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_k_dim, 2, dtype=torch.float32) / self.head_k_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotary(self, x):
        """x [B,T,H,D]; GPT-NeoX style rotate-half (fla RotaryEmbedding, interleaved=False)."""
        angles = torch.outer(torch.arange(x.shape[1], device=x.device, dtype=torch.float32), self.inv_freq)
        cos = torch.cat([angles.cos()] * 2, -1).to(x.dtype)[None, :, None]
        sin = torch.cat([angles.sin()] * 2, -1).to(x.dtype)[None, :, None]
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def forward(self, x):
        B, T, _ = x.shape
        q = self.rotary(self.q_proj(x).view(B, T, self.num_heads, self.head_k_dim))
        k = self.rotary(self.k_proj(x).view(B, T, self.num_heads, self.head_k_dim))
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_v_dim)
        o = retention(q, k, v, self.retention_backend)
        g = self.g_proj(x).view(B, T, self.num_heads, self.head_v_dim)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.g_norm_weight * F.silu(g)
        return self.o_proj(o.reshape(B, T, -1))


# ----------------------------------------------------------------------------
# Wrappers for vendored layers, and LSTM
# ----------------------------------------------------------------------------

class HyenaMixer(nn.Module):
    def __init__(self, d_model, l_max, filter_order):
        super().__init__()
        # lr_pos_emb=0 registers the filter positional features as a buffer: the trainers here have one
        # learning rate, so the official per-tensor lr 1e-5 cannot be honoured; freezing is closer to it.
        self.op = HyenaOperator(d_model, l_max, order=2, filter_order=filter_order, lr_pos_emb=0.0)

    def forward(self, x):
        return self.op(x)


class S4DMixer(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.layer = S4D(d_model, d_state=d_state, transposed=False)

    def forward(self, x):
        return self.layer(x)[0]


class LSTMMixer(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.rnn = nn.LSTM(d_model, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.out(self.rnn(x)[0])


def make_mixer(name, d_model, num_heads, l_max, width=None, retention_backend="auto"):
    """width=None uses the official internal width (OFFICIAL_WIDTH). retention_backend applies to retnet only."""
    if name in OFFICIAL_WIDTH and width is None:
        width = OFFICIAL_WIDTH[name](d_model)
    if name == "transformer":
        return CausalSelfAttention(d_model, num_heads)
    if name == "gqa":
        return GroupedQueryAttention(d_model, num_heads, max(1, num_heads // 2))
    if name == "linear_attention":
        return CausalLinearAttention(d_model, num_heads)
    if name == "gla":
        return GatedLinearAttention(d_model, num_heads)
    if name == "mamba":
        return MambaMixer(d_model, width)
    if name == "hyena":
        return HyenaMixer(d_model, l_max, width)
    if name == "lstm":
        return LSTMMixer(d_model, width)
    if name == "s4d":
        return S4DMixer(d_model, width)
    if name == "retnet":
        return RetNetMixer(d_model, num_heads, width, retention_backend=retention_backend)
    raise ValueError(f"unknown common-set mixer: {name}; known: {NAMES}")
