#!/usr/bin/env python3
"""
Tier 0 Zero-Cost Proxy harness  (Beyond-Transformer / KCVL)

학습 0회. init 상태의 모델을 6개 지표로 스코어링하고,
지표 간 Spearman 순위상관을 출력한다.

사용 (proxy_tasks/ 에서):
    python -m tier0.tier0_proxy --quick
    python -m tier0.tier0_proxy --seeds 3 --seq-len 256 --probe-len 24
결과는 --out 을 주지 않으면 $PROXY_OUTPUT_ROOT/tier0/<models>-s1234/tier0_results.json.

지표:
  synflow         SynFlow            (data-free, |theta| 치환 후 ones 입력)
  snip            SNIP               (|theta * dL/dtheta| 합)
  dss_pp          DSS++ (generalized) token-mixing 행렬 nuclear norm + MLP saliency
  naswot          NASWOT             (활성 패턴 Hamming 커널 logdet)
  ntk_cond        NTK condition num  (-log10(cond) 로 부호 정렬)
  state_cap       State capacity     (추론 시 상태 바이트, 해석적)
대조군:
  params, flops

의존성: torch 만. 외부 패키지 없음 (Mamba/GLA 등은 순수 PyTorch 재구현).

알려진 한계 (실행해 확인함. 이 수치를 그대로 인용하지 말 것):
  아래 수치는 서버에서 CUDA 초기화가 실패한 상태로 CPU 에서 돈 결과다
  (artifacts/tier0/main_b*.json, env.device == "cpu"). 같은 코드를 GPU 에서
  돌리면 값이 소수 셋째 자리부터 달라지고, LSTM 은 cudnn RNN 이 eval 모드
  backward 를 지원하지 않아 snip 이후 지표가 NaN 이 된다.
  - SynFlow: 입력이 전 위치 동일한 ones 텐서라 token mixing 차이가 점수에
    드러나지 않는다. gla 와 rwkv6 가 같은 값(207.58), transformer 와 swa 가
    거의 같은 값이 나왔다. mamba/h3/hyena 는 float64 에서도 오버플로우로 NaN 이다.
  - NASWOT: 훅이 MLP 의 SiLU 에만 걸려 mixer 를 거의 보지 못한다.
    11종 중 7종이 seed 표준편차(약 0.01) 안에 몰렸다.
  - dss_pp: 원 DSS++ 가 아니다. 원본은 attention 행렬이 필요해서 SSM 에 적용할
    수 없어 token-mixing 야코비안 nuclear norm 으로 대체한 자체 변형이다.
    논문에서 DSS++ 로 인용하면 안 된다.
  - ntk_cond: out.mean() 을 미분해 샘플당 야코비안이 1행뿐이라 degenerate 하다.
    표준 empirical NTK 가 아니다.
  - probe_len 이 window(64) 보다 짧으면 swa 가 transformer 와 동일해진다.
    probe_len 24 에서 둘 다 294.509 가 나왔다.
  - RWKV-6 의 gate 가 nn.Linear(d, n_head) 라 GLA 와 같은 head 단위 감쇠다.
    채널 단위가 아니다.
  - Based 가 2차 Taylor feature map 이 아니라 elu+1 + GLA 감쇠다.
"""

import argparse, json, math, os, platform, sys, time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.paths import task_output

# ----------------------------------------------------------------------------
# 환경 점검
# ----------------------------------------------------------------------------

def env_report(device: str) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_capability"] = [".".join(map(str, torch.cuda.get_device_capability(i)))
                                  for i in range(torch.cuda.device_count())]
        free, total = torch.cuda.mem_get_info()
        info["gpu0_mem_free_GB"] = round(free / 1024**3, 2)
        info["gpu0_mem_total_GB"] = round(total / 1024**3, 2)
    print("=" * 74)
    print("ENVIRONMENT")
    for k, v in info.items():
        print(f"  {k:22s} {v}")
    print("=" * 74)
    return info


# ----------------------------------------------------------------------------
# 공통 빌딩 블록
# ----------------------------------------------------------------------------

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
            # RWKV-6: token-shift 로 섞은 입력 + 채널 단위 감쇠
            self.mu = nn.Parameter(torch.rand(d))
            self.g = nn.Linear(d, n_head, bias=True)
            nn.init.constant_(self.g.bias, 1.0)

    def forward(self, x):
        B, L, D = x.shape
        if self.mode != "gla":
            prev = F.pad(x, (0, 0, 1, 0))[:, :L]
            x = x * self.mu + prev * (1 - self.mu)   # token-shift
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        q, k = F.elu(q) + 1, F.elu(k) + 1
        logg = -F.softplus(self.g(x)).transpose(1, 2)          # (B,h,L) <= 0
        c = logg.cumsum(-1)
        # j>i 에서 c_i-c_j > 0 이라 exp 가 inf 로 터지고,
        # masked_fill 의 0 기울기와 곱해져 backward 가 NaN 이 된다. max=0 클램프 필수.
        dec = (c[..., :, None] - c[..., None, :]).clamp(min=-30, max=0).exp()
        m = causal_mask(L, x.device)
        a = (q @ k.transpose(-1, -2)) * dec.masked_fill(~m, 0.0)
        a = a / (a.sum(-1, keepdim=True) + 1e-6)
        return self.o((a @ v).transpose(1, 2).reshape(B, L, D))


class Based(nn.Module):
    """Based: 2차 Taylor 선형 어텐션 + sliding window 를 합산."""
    kind = "linear_attn"

    def __init__(self, d, n_head=4, window=64):
        super().__init__()
        self.lin = LinearAttnDecay(d, n_head)
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

    def forward(self, idx=None, x=None):
        """idx: 토큰 id. x: 임베딩 우회 입력 (SynFlow 용)."""
        h = self.emb(idx) if x is None else x
        for b in self.blocks:
            h = b(h)
        return self.head(self.nf(h))


MIXERS = {
    "transformer": (lambda d: Attention(d, 4), "attention"),
    "swa":         (lambda d: Attention(d, 4, window=64), "attention_window"),
    "gla":         (lambda d: LinearAttnDecay(d, 4, "gla"), "linear_attn"),
    "rwkv6":       (lambda d: LinearAttnDecay(d, 4, "rwkv"), "linear_attn"),
    "based":       (lambda d: Based(d, 4, window=64), "linear_attn+window"),
    "s4d":         (lambda d: DiagonalSSM(d, 16), "ssm"),
    "mamba":       (lambda d: Mamba(d, 16), "ssm_selective"),
    "h3":          (lambda d: H3(d, 16), "ssm_gated"),
    "hyena":       (lambda d: Hyena(d), "conv_implicit"),
    "lstm":        (lambda d: LSTMMixer(d), "rnn"),
    "cnn":         (lambda d: ConvMixer(d), "conv"),
}


def build(name, cfg, device):
    fn, _ = MIXERS[name]
    m = LM(cfg.vocab, cfg.d_model, cfg.n_layers, lambda: fn(cfg.d_model)).to(device)
    return m


# ----------------------------------------------------------------------------
# 지표
# ----------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def flops_proxy(model, cfg):
    """forward FLOPs 근사. 선형층 2*in*out*L + 어텐션 2*L^2*d."""
    L, d = cfg.seq_len, cfg.d_model
    f = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            f += 2 * m.in_features * m.out_features * L
        elif isinstance(m, nn.Conv1d):
            f += 2 * m.in_channels * m.out_channels // m.groups * m.kernel_size[0] * L
        elif isinstance(m, nn.LSTM):
            f += 2 * 4 * m.input_size * m.hidden_size * L * 2
        elif isinstance(m, Attention):
            w = m.window if m.window is not None else L
            f += 2 * 2 * L * min(L, w) * d
    return f


def state_capacity_bytes(name, cfg, dtype_bytes=2):
    """추론 시 유지해야 하는 상태 크기(바이트). 해석적."""
    L, d, n_l, n_state, win, filt = cfg.seq_len, cfg.d_model, cfg.n_layers, 16, 64, 64
    if name == "transformer":
        s = 2 * L * d
    elif name == "swa":
        s = 2 * win * d
    elif name in ("gla", "rwkv6"):
        s = d * (d // 4)
    elif name == "based":
        s = d * (d // 4) + 2 * win * d
    elif name in ("s4d", "mamba", "h3"):
        s = d * n_state
    elif name == "hyena":
        s = 2 * filt * d
    elif name == "cnn":
        s = 5 * 4 * d
    elif name == "lstm":
        s = 2 * d
    else:
        raise KeyError(name)
    return s * n_l * dtype_bytes


@torch.no_grad()
def _linearize(model):
    signs = {}
    for n, p in model.state_dict().items():
        signs[n] = torch.sign(p)
        p.abs_()
    return signs


@torch.no_grad()
def _restore(model, signs):
    for n, p in model.state_dict().items():
        p.mul_(signs[n])


class _BypassNorm:
    """SynFlow 선형화 구간에서 RMSNorm 을 항등으로 둔다.
    정규화를 켠 채로 돌리면 신호 크기가 매 블록에서 재설정돼
    점수가 아키텍처와 무관하게 거의 상수가 된다(관측됨)."""
    def __enter__(self):
        self.orig = RMSNorm.forward
        RMSNorm.forward = lambda self_, x: x
        return self

    def __exit__(self, *a):
        RMSNorm.forward = self.orig


def synflow(model, cfg, device):
    """data-free. |theta| 치환 후 ones 입력, sum(theta * grad).
    정규화를 끄면 깊이에 따라 크기가 지수로 커져 float32 에서 inf 가 난다
    (mamba/h3/hyena 에서 관측). 원 구현대로 float64 로 돌린다."""
    model.double()
    signs = _linearize(model)
    model.zero_grad(set_to_none=True)
    x = torch.ones(1, cfg.seq_len, cfg.d_model, device=device, dtype=torch.float64)
    try:
        with _BypassNorm():
            out = model(x=x)
        out.sum().backward()
        score, n_ok, n_bad = 0.0, 0, 0
        for p in model.parameters():
            if p.grad is not None:
                g = (p * p.grad).sum()
                if torch.isfinite(g):
                    score += g.item()
                    n_ok += 1
                else:
                    n_bad += 1
        # 기여분이 전부 비유한이면 점수 0 이 되는데, 이를 log 하면 바닥값(-27.6)이
        # 되어 "매우 낮은 점수"처럼 보인다. 측정 실패이므로 NaN 으로 구분한다.
        if n_ok == 0 or score == 0.0:
            val = float("nan")
        else:
            val = math.log(abs(score))
        if n_bad:
            synflow.last_overflow = n_bad
    except Exception:
        val = float("nan")
    model.zero_grad(set_to_none=True)
    _restore(model, signs)
    model.float()
    return val


def snip(model, batch, device):
    model.zero_grad(set_to_none=True)
    idx, tgt = batch
    logits = model(idx)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
    loss.backward()
    score = sum((p * p.grad).abs().sum().item() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return math.log(score + 1e-12)


def naswot(model, batch, device):
    """활성 패턴의 Hamming 커널 logdet."""
    codes = []
    hooks = []

    def hook(_m, _i, o):
        codes.append((o > 0).flatten(1).float())

    for m in model.modules():
        if isinstance(m, nn.SiLU):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(batch[0])
    for h in hooks:
        h.remove()
    if not codes:
        return float("nan")
    C = torch.cat(codes, 1)
    n, f = C.shape
    K = C @ C.t() + (1 - C) @ (1 - C).t()
    K = K / f
    K = K + 1e-4 * torch.eye(n, device=K.device)
    return torch.linalg.slogdet(K)[1].item()


def ntk_cond(model, batch, device, n_sample=8):
    """K = J J^T 의 조건수. 작을수록 좋으므로 -log10 으로 부호 정렬."""
    idx = batch[0][:n_sample]
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(idx.size(0)):
        model.zero_grad(set_to_none=True)
        out = model(idx[i : i + 1])
        g = torch.autograd.grad(out.mean(), params, retain_graph=False, allow_unused=True)
        rows.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten()
                               for gi, p in zip(g, params)]))
    J = torch.stack(rows).double()
    if not torch.isfinite(J).all():
        return float("nan")      # 모델 forward/backward 자체가 깨진 경우 — 조용히 숨기지 않는다
    # eigvalsh(J J^T) 는 조건수가 큰 행렬에서 수렴 실패한다.
    # J 의 특이값을 직접 구하면 제곱 없이 같은 값을 안정적으로 얻는다.
    try:
        s = torch.linalg.svdvals(J).clamp(min=1e-20)
        cond = ((s[0] / s[-1]) ** 2).item()
    except Exception:
        K = J @ J.t()
        K = K + 1e-10 * torch.eye(K.size(0), dtype=K.dtype, device=K.device) * K.diagonal().mean()
        ev = torch.linalg.eigvalsh(K).clamp(min=1e-20)
        cond = (ev[-1] / ev[0]).item()
    model.zero_grad(set_to_none=True)
    return -math.log10(cond + 1e-12)


def dss_pp(model, cfg, device, probe_len=32):
    """DSS++ 일반화판.
    원 논문은 MSA 의 attention 행렬 nuclear norm(synaptic diversity)을 쓰지만,
    SSM/conv 에는 attention 행렬이 없다. 대신 모든 mixer 에 대해
    '경험적 token-mixing 야코비안' T (Lp x Lp) 를 뽑아 nuclear norm 을 쓴다.
    MLP 쪽은 원 논문대로 saliency.
    """
    total = 0.0
    for blk in model.blocks:
        x = torch.randn(1, probe_len, cfg.d_model, device=device, requires_grad=True)
        y = blk.mixer(blk.n1(x))
        T = torch.zeros(probe_len, probe_len, device=device)
        for i in range(probe_len):
            g = torch.autograd.grad(y[0, i].sum(), x, retain_graph=True)[0]
            T[i] = g[0].abs().sum(-1)
        nuc = torch.linalg.matrix_norm(T, ord="nuc").item() / probe_len
        sal = sum(p.abs().sum().item() for p in blk.mlp.parameters())
        total += nuc * math.log(sal + 1e-12)
    return total


# ----------------------------------------------------------------------------
# Spearman
# ----------------------------------------------------------------------------

def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return float("nan")
    ra, rb = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


# ----------------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------------

@dataclass
class Cfg:
    vocab: int = 256
    d_model: int = 128
    n_layers: int = 4
    seq_len: int = 256
    batch: int = 8


PROXIES = ["synflow", "snip", "dss_pp", "naswot", "ntk_cond", "state_cap"]
BASELINES = ["params", "flops"]


def run_one(name, cfg, device, seed, probe_len):
    torch.manual_seed(seed)
    model = build(name, cfg, device)
    model.eval()
    idx = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq_len), device=device)
    batch = (idx, idx)
    r = {"params": float(count_params(model)), "flops": float(flops_proxy(model, cfg))}
    r["state_cap"] = float(state_capacity_bytes(name, cfg))
    r["synflow"] = synflow(model, cfg, device)
    r["snip"] = snip(model, batch, device)
    r["naswot"] = naswot(model, batch, device)
    r["ntk_cond"] = ntk_cond(model, batch, device)
    r["dss_pp"] = dss_pp(model, cfg, device, probe_len=probe_len)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--probe-len", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None,
                    help="기본: $PROXY_OUTPUT_ROOT/tier0/<models>-s1234/tier0_results.json (1234 = 첫 seed)")
    ap.add_argument("--quick", action="store_true", help="스모크 테스트용 축소 설정")
    a = ap.parse_args()

    if a.quick:
        a.seeds, a.d_model, a.n_layers, a.seq_len, a.batch, a.probe_len = 1, 32, 2, 64, 4, 8
    if a.out is None:
        a.out = str(task_output("tier0", a.models.replace(",", "+"), 1234) / "tier0_results.json")

    cfg = Cfg(d_model=a.d_model, n_layers=a.n_layers, seq_len=a.seq_len, batch=a.batch)
    names = list(MIXERS) if a.models == "all" else a.models.split(",")

    env = env_report(a.device)
    print(f"CONFIG  d_model={cfg.d_model} n_layers={cfg.n_layers} L={cfg.seq_len} "
          f"B={cfg.batch} seeds={a.seeds} probe_len={a.probe_len}")
    if cfg.seq_len <= 64 and ("swa" in names or "based" in names):
        print("  [주의] window=64 >= L 이라 swa 가 transformer 와 동일해진다. "
              "L>=256 으로 돌릴 것.")
    print(f"MODELS  {len(names)}종: {', '.join(names)}\n")

    results, t0 = {}, time.time()
    for name in names:
        per_seed = []
        t1 = time.time()
        for s in range(a.seeds):
            try:
                per_seed.append(run_one(name, cfg, a.device, 1234 + s, a.probe_len))
            except Exception as e:
                print(f"  [FAIL] {name} seed={s}: {type(e).__name__}: {e}")
                per_seed.append({k: float("nan") for k in PROXIES + BASELINES})
        keys = PROXIES + BASELINES
        mean = {k: sum(d[k] for d in per_seed) / len(per_seed) for k in keys}
        std = {k: (sum((d[k] - mean[k]) ** 2 for d in per_seed) / len(per_seed)) ** 0.5 for k in keys}
        results[name] = {"mean": mean, "std": std, "seeds": per_seed,
                         "family": MIXERS[name][1]}
        print(f"  {name:12s} {time.time()-t1:6.1f}s  " +
              "  ".join(f"{k}={mean[k]:.3g}" for k in PROXIES))

    # 점수표
    print("\n" + "=" * 74)
    print("SCORES (seed 평균)")
    hdr = f"{'model':12s}" + "".join(f"{k:>13s}" for k in PROXIES + BASELINES)
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        m = results[n]["mean"]
        print(f"{n:12s}" + "".join(f"{m[k]:13.4g}" for k in PROXIES + BASELINES))

    # 순위
    print("\nRANKING (각 지표에서 1위가 가장 높은 점수)")
    for k in PROXIES + BASELINES:
        vals = [(n, results[n]["mean"][k]) for n in names]
        vals.sort(key=lambda t: (-t[1] if not math.isnan(t[1]) else float("inf")))
        print(f"  {k:11s} " + " > ".join(n for n, _ in vals))

    # 지표 간 Spearman
    print("\n" + "=" * 74)
    print("SPEARMAN rho (지표 간 순위상관). |rho|>0.9 면 중복 의심.")
    allk = PROXIES + BASELINES
    print(f"{'':12s}" + "".join(f"{k[:9]:>10s}" for k in allk))
    for i, ki in enumerate(allk):
        row = f"{ki:12s}"
        for kj in allk:
            r = spearman([results[n]["mean"][ki] for n in names],
                         [results[n]["mean"][kj] for n in names])
            row += f"{r:10.2f}" if not math.isnan(r) else f"{'nan':>10s}"
        print(row)

    payload = {"env": env, "config": cfg.__dict__, "seeds": a.seeds,
               "probe_len": a.probe_len, "results": results}
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n총 {time.time()-t0:.1f}s · 저장: {a.out}")


if __name__ == "__main__":
    main()
