import numpy as np
import torch
from torch.nn import functional as F
from .base import ArrayCls, Part, batched

class GenPart(Part):
    """Parts whose data is generated on the fly (training) and from fixed CPU-seeded sets (evaluation)."""
    eval_bs = 250
    gen_on_cpu = True
    chance = ceiling = None

    def cpu_gen(self, seed):
        return torch.Generator().manual_seed(seed)

class MadCompression(ArrayCls):
    """MAD compression (Poli et al. 2024): vocab 16, length 32, last token is the compression token (15); the model
    encodes the sequence into the last position's vector and an MLP decoder reconstructs all 32 tokens. The MAD cache
    t-C_vs-16_sl-32_ntr-12800 is reused: its first 11,520 train rows train, its last 1,280 train rows validate
    (the cache's own test split is not used)."""
    name = 'mad_compression'
    in_kind, in_dim, out_dim, readout, l_max, ae_len = ('tok', 16, 16, 'ae', 32, 32)
    batch_size = 128
    score_name = 'token_accuracy'
    cache = 't-C_vs-16_sl-32_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-0_s-12345'
    n_val = 1280

    @property
    def path(self):
        return self.data_dir

    def _arrays(self):
        x = np.load(self.path / 'inputs.npy')
        return (x[:-self.n_val], x[-self.n_val:])

    def load(self, device):
        tr, va = self._arrays()
        self.xtr, self.xva = (torch.from_numpy(tr).to(device), torch.from_numpy(va).to(device))
        self.ytr, self.yva = (self.xtr, self.xva)

    def sample(self, g):
        i = torch.randint(self.xtr.shape[0], (self.batch_size,), generator=g, device=self.xtr.device)
        return {'x': self.xtr[i], 'y': self.xtr[i]}

    def loss(self, out, batch):
        return F.cross_entropy(out.reshape(-1, 16), batch['y'].reshape(-1))

    def _acc(self, model, x):
        out = batched(model, lambda lo, hi: (x[lo:hi], None), x.shape[0], 256)
        return (out.argmax(-1) == x).float().mean().item()

    def evaluate(self, model):
        return {'score': self._acc(model, self.xva)}

    def evaluate_train(self, model):
        return {'score': self._acc(model, self.xtr[:2000])}

    def baseline(self):
        tr, va = self._arrays()
        mode = np.array([np.bincount(tr[:, t], minlength=16).argmax() for t in range(32)])
        n_data = 15
        return {'baseline': float((va == mode[None]).mean()), 'chance': (31 / n_data + 1) / 32, 'ceiling': 1.0, 'vocab_data_tokens': n_data, 'train_rows': int(tr.shape[0]), 'val_rows': int(va.shape[0])}
