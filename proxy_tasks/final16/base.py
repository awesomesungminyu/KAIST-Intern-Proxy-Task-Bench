import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..common_set.backbone import Backbone, RMSNorm
DATA = Path(os.environ.get('SMOKE_V21_DATA', str(Path(os.environ.get('PROXY_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data')) / 'smoke_v21')))

def use_pylib():
    """pyarrow lives in DATA/_pylib (installed with pip --target so the shared conda env stays untouched)."""
    lib = str(DATA / '_pylib')
    if lib not in sys.path:
        sys.path.append(lib)

def sincos_1d(length, dim):
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    pe = torch.zeros(length, dim)
    pe[:, 0::2], pe[:, 1::2] = (torch.sin(pos * freq), torch.cos(pos * freq))
    return pe

class SeqModel(nn.Module):
    """read-in -> common-set Backbone -> read-out.

    in_kind   'tok' (Embedding) or 'vec' (Linear)
    readout   'all' per-position logits [B,T,out]; 'last' [B,out]; 'mean' masked time mean [B,out];
              'ae'  MAD autoencoder read-out: last-position vector -> positional MLP decoder -> [B,ae_len,out]
    """

    def __init__(self, in_kind, in_dim, out_dim, readout, mixer, d_model, n_layers, n_heads, l_max, ae_len=None):
        super().__init__()
        self.readout = readout
        self.embed = nn.Embedding(in_dim, d_model) if in_kind == 'tok' else nn.Linear(in_dim, d_model)
        self.backbone = Backbone(mixer, d_model, n_layers, n_heads, l_max)
        if readout == 'ae':
            self.register_buffer('dec_pos', sincos_1d(ae_len, d_model), persistent=False)
            self.decoder = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), RMSNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
            self.head = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, out_dim))
        else:
            self.head = nn.Linear(d_model, out_dim, bias=readout != 'all')

    def forward(self, x, mask=None):
        h = self.backbone(self.embed(x))
        if self.readout == 'all':
            return self.head(h)
        if self.readout == 'last':
            return self.head(h[:, -1])
        if self.readout == 'mean':
            if mask is None:
                return self.head(h.mean(1))
            m = mask.unsqueeze(-1).to(h.dtype)
            return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))
        z = h[:, -1]
        return self.head(self.decoder(z.unsqueeze(1) + self.dec_pos.to(z.dtype)[None]))

    def num_params(self):
        return sum((p.numel() for p in self.parameters()))

class Part:
    """One trained model + one metric. Subclasses set the class attributes and implement the methods below."""
    name = 'main'
    in_kind = 'tok'
    in_dim = 0
    out_dim = 0
    readout = 'mean'
    l_max = 0
    ae_len = None
    d_model, n_layers, n_heads = (128, 2, 4)
    batch_size = 64
    steps = 1000
    lr = 0.001
    wd = 0.1
    warmup_frac = 0.05
    clip = 1.0
    score_name = 'accuracy'
    direction = 'max'
    pad_id = None
    kind = 'cls'
    gen_on_cpu = False

    def prepare(self):
        """CPU-side, idempotent: download/process into DATA (not part of any cell's wall time)."""

    def load(self, device):
        """Move data to the GPU."""

    def sample(self, g):
        """-> batch dict (device tensors); g is a torch.Generator on the device."""
        raise NotImplementedError

    def inputs(self, batch):
        x = batch['x']
        return (x, x != self.pad_id if self.pad_id is not None else None)

    def loss(self, out, batch):
        raise NotImplementedError

    def evaluate(self, model):
        """-> dict with at least 'score' (primary, validation)."""
        raise NotImplementedError

    def evaluate_train(self, model):
        return None

    def baseline(self):
        """Data-only quantities: dict(baseline=..., chance=..., plus anything useful). Never uses a model."""
        raise NotImplementedError

    def describe(self):
        """Facts about the data actually used in this run (stored in result.json)."""
        return {}

    def build(self, mixer):
        return SeqModel(self.in_kind, self.in_dim, self.out_dim, self.readout, mixer, self.d_model, self.n_layers, self.n_heads, self.l_max, self.ae_len)

@torch.no_grad()
def batched(model, fn, n, bs):
    """Run fn(lo, hi) -> (x, mask) through the model in chunks and return the concatenated outputs."""
    outs = []
    for lo in range(0, n, bs):
        x, mask = fn(lo, min(n, lo + bs))
        outs.append(model(x, mask))
    return torch.cat(outs)

class ArrayCls(Part):
    """Classification on in-memory arrays: self.xtr/ytr/xva/yva set by load(); mean read-out unless overridden."""
    readout = 'mean'
    eval_bs = 256
    train_diag_n = 2000
    trim = False

    def _x(self, x):
        x = x.float() if self.in_kind == 'vec' else x.long()
        if self.trim:
            x = x[:, :max(1, int((x != 0).sum(1).max()))]
        return x

    def sample(self, g):
        i = torch.randint(self.xtr.shape[0], (self.batch_size,), generator=g, device=self.xtr.device)
        return {'x': self._x(self.xtr[i]), 'y': self.ytr[i].long()}

    def loss(self, out, batch):
        return F.cross_entropy(out, batch['y'])

    def _acc(self, model, x, y):

        def fn(lo, hi):
            xb = self._x(x[lo:hi])
            return (xb, xb != self.pad_id if self.pad_id is not None else None)
        out = batched(model, fn, x.shape[0], self.eval_bs)
        return (out.argmax(-1) == y.long()).float().mean().item()

    def evaluate(self, model):
        return {'score': self._acc(model, self.xva, self.yva)}

    def evaluate_train(self, model):
        n = min(self.train_diag_n, self.xtr.shape[0])
        return {'score': self._acc(model, self.xtr[:n], self.ytr[:n])}

    def majority(self, ytr, yva):
        return float((np.bincount(np.asarray(ytr)).argmax() == np.asarray(yva)).mean())
