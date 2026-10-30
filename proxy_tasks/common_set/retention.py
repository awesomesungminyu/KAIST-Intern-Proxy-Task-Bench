"""Retention kernels for the RetNet mixer.

CUDA uses flash-linear-attention 0.3.1 (MIT), installed with bash setup.sh retnet.
CPU uses the equivalent PyTorch reference for contract tests. CUDA kernel failures are reported explicitly.
Set PROXY_FLA_VENDOR only to relocate the optional vendor directory.
"""
import functools
import os
import sys
import types
from pathlib import Path

import torch


def retention_reference(q, k, v):
    """q,k [B,T,H,K], v [B,T,H,V] -> [B,T,H,V].  o_t = sum_{s<=t} gamma_h^(t-s) (q_t . k_s) K^-0.5 v_s,
    gamma_h = 1 - 2^(-5-h), the decay chunk_retention derives from the head index."""
    T, H, K = q.shape[1], q.shape[2], q.shape[3]
    log_gamma = torch.log1p(-torch.pow(2.0, -5.0 - torch.arange(H, device=q.device, dtype=torch.float32)))
    pos = torch.arange(T, device=q.device)
    diff = (pos[:, None] - pos[None, :]).to(torch.float32)
    causal = diff >= 0
    decay = torch.exp(log_gamma[:, None, None] * diff.clamp(min=0)) * causal    # [H,T,T]
    scores = torch.einsum("bthk,bshk->bhts", q * K ** -0.5, k) * decay.to(q.dtype)
    return torch.einsum("bhts,bshv->bthv", scores, v)


def fla_vendor_root():
    path = os.environ.get("PROXY_FLA_VENDOR") or str(Path(__file__).resolve().parents[2] / "third_party/fla_vendor")
    return Path(path).expanduser().resolve()


@functools.lru_cache(maxsize=1)
def fla_chunk_retention():
    root = fla_vendor_root()
    if not (root / "fla").is_dir():
        raise RuntimeError(f"fla not found at {root}; run bash setup.sh retnet")
    if str(root) not in sys.path:
        sys.path.append(str(root))     # after the environment, so its own packages win
    if "fla" not in sys.modules:
        package = types.ModuleType("fla")
        package.__path__ = [str(root / "fla")]
        sys.modules["fla"] = package
    from fla.ops.retention import chunk_retention
    return chunk_retention


BACKENDS = ("auto", "torch")


def retention(q, k, v, backend="auto"):
    """backend "auto": fla chunk_retention on CUDA, retention_reference on CPU; "torch": retention_reference."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown retention backend {backend!r}; known: {BACKENDS}")
    if q.is_cuda and backend == "auto":
        try:
            kernel = fla_chunk_retention()
        except Exception as exc:
            raise RuntimeError(f"retnet requires fla chunk_retention on CUDA (torch {torch.__version__}): {exc}") from exc
        return kernel(q=q, k=k, v=v)[0]
    return retention_reference(q, k, v)
