"""Tier0 mixer contract: float (B,L,d) -> float (B,L,d), causal, no carried state.

Register a builder(d) with family and state_elements(cfg). The latter is an
analytic estimate per layer; the evaluator multiplies by layers and 2 bytes.
LM/Block and all mixer classes preserve the v2 implementation and initialization.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from common.registry import ModelRegistry

MODELS = ModelRegistry("tier0")
MODEL_INFO = {}


def register_model(name, *, family, state_elements):
    def decorate(builder):
        MODELS.register(name)(builder)
        MODEL_INFO[name] = {"family": family, "state_elements": state_elements}
        return builder
    return decorate


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MLP(nn.Module):
    """SwiGLU. act 를 nn.Module 로 둬야 NASWOT 훅이 걸린다."""
    def __init__(self, d, mult=4):
        super().__init__()
        h = int(d * mult * 2 / 3 / 8) * 8
        self.up = nn.Linear(d, h, bias=False)
        self.gate = nn.Linear(d, h, bias=False)
        self.act = nn.SiLU()
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


def causal_mask(L, device):
    return torch.ones(L, L, dtype=torch.bool, device=device).tril()


# ----------------------------------------------------------------------------
# Token mixer 11종
# ----------------------------------------------------------------------------

class Attention(nn.Module):
    """Transformer++ : causal MHA"""
    kind = "attention"

    def __init__(self, d, n_head=4, window=None):
        super().__init__()
        self.h, self.dh, self.window = n_head, d // n_head, window
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        m = causal_mask(L, x.device)
        if self.window is not None:
            idx = torch.arange(L, device=x.device)
            m = m & ((idx[:, None] - idx[None, :]) < self.window)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        a = a.masked_fill(~m, float("-inf")).softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, L, D))


class LinearAttnDecay(nn.Module):
    """GLA / RWKV-6 계열. 게이트 누적감쇠를 가진 선형 어텐션.
    D[i,j] = exp(cumlogg_i - cumlogg_j) (j<=i) 로 O(L^2) 벡터화."""
    kind = "linear_attn"

    def __init__(self, d, n_head=4, gate_mode="gla"):
        super().__init__()
        self.h, self.dh, self.mode = n_head, d // n_head, gate_mode
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if gate_mode == "gla":
            # GLA: 입력 의존 head 단위 게이트
            self.g = nn.Linear(d, n_head, bias=True)
            nn.init.constant_(self.g.bias, 2.0)
        else:
            # RWKV-6: token-shift 로 섞은 입력 + 입력 의존 채널(key 차원) 단위 감쇠
            self.mu = nn.Parameter(torch.rand(d))
            self.g = nn.Linear(d, d, bias=True)
            nn.init.constant_(self.g.bias, 1.0)

    def forward(self, x):
        B, L, D = x.shape
        if self.mode != "gla":
            prev = F.pad(x, (0, 0, 1, 0))[:, :L]
            x = x * self.mu + prev * (1 - self.mu)   # token-shift
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        q, k = F.elu(q) + 1, F.elu(k) + 1
        m = causal_mask(L, x.device)
        # j>i 에서 c_i-c_j > 0 이라 exp 가 inf 로 터지고,
        # masked_fill 의 0 기울기와 곱해져 backward 가 NaN 이 된다. max=0 클램프 필수.
        if self.mode == "gla":
            logg = -F.softplus(self.g(x)).transpose(1, 2)      # (B,h,L) <= 0
            c = logg.cumsum(-1)
            dec = (c[..., :, None] - c[..., None, :]).clamp(min=-30, max=0).exp()
            a = (q @ k.transpose(-1, -2)) * dec.masked_fill(~m, 0.0)
        else:
            logg = -F.softplus(self.g(x)).view(B, L, self.h, self.dh).transpose(1, 2)  # (B,h,L,dh)
            c = logg.cumsum(2)
            dec = (c[:, :, :, None] - c[:, :, None, :]).clamp(min=-30, max=0).exp()   # (B,h,L,L,dh)
            a = torch.einsum("bhic,bhjc,bhijc->bhij", q, k, dec).masked_fill(~m, 0.0)
        a = a / (a.sum(-1, keepdim=True) + 1e-6)
        return self.o((a @ v).transpose(1, 2).reshape(B, L, D))


class TaylorLinearAttn(nn.Module):
    """Based 의 선형 어텐션 부분. 감쇠 없음.
    phi(x) = [1, x, x⊗x/sqrt(2)] 이면 phi(q)·phi(k) = 1 + s + s^2/2 (s = q·k) 라
    O(L^2) 경로에선 feature 를 펼치지 않고 커널을 직접 계산한다."""
    kind = "linear_attn"

    def __init__(self, d, n_head=4, d_feat=16):
        super().__init__()
        self.h, self.dh, self.df = n_head, d // n_head, d_feat
        self.qk = nn.Linear(d, 2 * n_head * d_feat, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k = self.qk(x).chunk(2, -1)
        q, k = (t.view(B, L, self.h, self.df).transpose(1, 2) for t in (q, k))
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.df)
        a = (1 + s + 0.5 * s * s).masked_fill(~causal_mask(L, x.device), 0.0)
        a = a / (a.sum(-1, keepdim=True) + 1e-6)
        return self.o((a @ v).transpose(1, 2).reshape(B, L, D))


class Based(nn.Module):
    """Based: 2차 Taylor 선형 어텐션 + sliding window 를 합산."""
    kind = "linear_attn"

    def __init__(self, d, n_head=4, window=64):
        super().__init__()
        self.lin = TaylorLinearAttn(d, n_head)
        self.swa = Attention(d, n_head, window=window)
        self.mix = nn.Linear(2 * d, d, bias=False)

    def forward(self, x):
        return self.mix(torch.cat([self.lin(x), self.swa(x)], -1))


class DiagonalSSM(nn.Module):
    """S4D: 대각 SSM 을 FFT 합성곱으로."""
    kind = "ssm"

    def __init__(self, d, d_state=16):
        super().__init__()
        self.d, self.n = d, d_state
        self.log_dt = nn.Parameter(torch.rand(d) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.log_A = nn.Parameter(torch.log(0.5 + torch.arange(d_state).float()).repeat(d, 1))
        self.C = nn.Parameter(torch.randn(d, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d, bias=False)

    def kernel(self, L, device):
        dt = self.log_dt.exp().unsqueeze(-1)                   # (d,1)
        A = -self.log_A.exp()                                  # (d,n)
        t = torch.arange(L, device=device).view(1, 1, L)
        k = (self.C.unsqueeze(-1) * torch.exp(A.unsqueeze(-1) * dt.unsqueeze(-1) * t)).sum(1)
        return k                                               # (d,L)

    def forward(self, x):
        B, L, D = x.shape
        k = self.kernel(L, x.device)
        n = 2 * L
        y = torch.fft.irfft(torch.fft.rfft(x.transpose(1, 2), n=n) * torch.fft.rfft(k, n=n), n=n)[..., :L]
        y = y.transpose(1, 2) + x * self.D
        return self.out(y)


class Mamba(nn.Module):
    """선택적 SSM. 순차 스캔(L 루프) — init 스코어링엔 충분."""
    kind = "ssm"

    def __init__(self, d, d_state=16, d_conv=4):
        super().__init__()
        self.d, self.n = d, d_state
        self.inp = nn.Linear(d, 2 * d, bias=False)
        self.conv = nn.Conv1d(d, d, d_conv, groups=d, padding=d_conv - 1)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(d, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d, 1))
        self.D = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.inp(x)
        u, z = xz.chunk(2, -1)
        u = self.act(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dbc = self.x_proj(u)
        dt = F.softplus(self.dt_proj(dbc[..., :1]))            # (B,L,d)
        Bm, Cm = dbc[..., 1:1 + self.n], dbc[..., 1 + self.n:]
        A = -self.A_log.exp()                                  # (d,n)
        h = x.new_zeros(B, D, self.n)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)         # (B,d,n)
            dB = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)
            h = dA * h + dB * u[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + u * self.D
        return self.out(y * self.act(z))


class H3(nn.Module):
    """shift SSM -> diagonal SSM, 게이팅."""
    kind = "ssm"

    def __init__(self, d, d_state=16):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.shift = nn.Conv1d(d, d, 3, groups=d, padding=2)
        self.ssm = DiagonalSSM(d, d_state)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        k = self.shift(k.transpose(1, 2))[..., :L].transpose(1, 2)
        return self.out(q * self.ssm(k * v))


class Hyena(nn.Module):
    """암묵적 장거리 합성곱 + 곱셈 게이팅."""
    kind = "conv"

    def __init__(self, d, order=2, filt=64):
        super().__init__()
        self.order, self.filt = order, filt
        self.proj = nn.Linear(d, (order + 1) * d, bias=False)
        self.pos = nn.Parameter(torch.randn(order, filt, d) / math.sqrt(filt))
        self.short = nn.Conv1d(d, d, 3, groups=d, padding=2)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        parts = self.proj(x).chunk(self.order + 1, -1)
        y = parts[0]
        for i in range(self.order):
            h = self.pos[i].transpose(0, 1)                    # (d,filt)
            n = 2 * L
            yf = torch.fft.rfft(y.transpose(1, 2), n=n) * torch.fft.rfft(h, n=n)
            y = torch.fft.irfft(yf, n=n)[..., :L].transpose(1, 2)
            y = y * parts[i + 1]
        y = self.short(y.transpose(1, 2))[..., :L].transpose(1, 2)
        return self.out(y)


class LSTMMixer(nn.Module):
    kind = "rnn"

    def __init__(self, d):
        super().__init__()
        self.rnn = nn.LSTM(d, d, batch_first=True)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        # cudnn RNN 은 eval 모드 backward 를 지원하지 않는다.
        with torch.backends.cudnn.flags(enabled=False):
            return self.out(self.rnn(x)[0])


class ConvMixer(nn.Module):
    """dilated depthwise CNN 대조군."""
    kind = "conv"

    def __init__(self, d, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, k, groups=d, padding=(k - 1))
        self.c2 = nn.Conv1d(d, d, k, groups=d, dilation=4, padding=4 * (k - 1))
        self.act = nn.SiLU()
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        y = self.c1(x.transpose(1, 2))[..., :L]
        y = self.act(y)
        y = self.c2(y)[..., :L]
        return self.out(y.transpose(1, 2))


# ----------------------------------------------------------------------------
# 모델 래퍼
# ----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d, mixer):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mixer, self.mlp = mixer, MLP(d)

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    def __init__(self, vocab, d, n_layers, mixer_fn):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, mixer_fn()) for _ in range(n_layers)])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        h = self.emb(idx)
        for b in self.blocks:
            h = b(h)
        return self.head(self.nf(h))


@register_model("transformer", family="attention", state_elements=lambda c: 2 * c.seq_len * c.d_model)
def _transformer(d):
    return Attention(d, 4)


@register_model("swa", family="attention_window", state_elements=lambda c: 2 * 64 * c.d_model)
def _swa(d):
    return Attention(d, 4, window=64)


@register_model("gla", family="linear_attn", state_elements=lambda c: c.d_model * (c.d_model // 4))
def _gla(d):
    return LinearAttnDecay(d, 4, "gla")


@register_model("rwkv6", family="linear_attn", state_elements=lambda c: c.d_model * (c.d_model // 4))
def _rwkv6(d):
    return LinearAttnDecay(d, 4, "rwkv")


@register_model("based", family="linear_attn+window", state_elements=lambda c: c.d_model * (c.d_model // 4) + 2 * 64 * c.d_model)
def _based(d):
    return Based(d, 4, window=64)


@register_model("s4d", family="ssm", state_elements=lambda c: c.d_model * 16)
def _s4d(d):
    return DiagonalSSM(d, 16)


@register_model("mamba", family="ssm_selective", state_elements=lambda c: c.d_model * 16)
def _mamba(d):
    return Mamba(d, 16)


@register_model("h3", family="ssm_gated", state_elements=lambda c: c.d_model * 16)
def _h3(d):
    return H3(d, 16)


@register_model("hyena", family="conv_implicit", state_elements=lambda c: 2 * 64 * c.d_model)
def _hyena(d):
    return Hyena(d)


@register_model("lstm", family="rnn", state_elements=lambda c: 2 * c.d_model)
def _lstm(d):
    return LSTMMixer(d)


@register_model("cnn", family="conv", state_elements=lambda c: 5 * 4 * c.d_model)
def _cnn(d):
    return ConvMixer(d)


def build(name, cfg, device):
    return LM(cfg.vocab, cfg.d_model, cfg.n_layers,
              lambda: MODELS.build(name, d=cfg.d_model)).to(device)


def state_capacity_bytes(name, cfg, dtype_bytes=2):
    """Analytic state estimate, retaining the v2 formulas (not measured memory)."""
    return MODEL_INFO[name]["state_elements"](cfg) * cfg.n_layers * dtype_bytes
