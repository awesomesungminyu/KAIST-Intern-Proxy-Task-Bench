"""Permuted Kuzushiji-49; 5,000 validation images, split seed 1, permutation seed 0."""
import numpy as np
import torch
from .base import ArrayCls
from .real import split_idx

class Kuzushiji49(ArrayCls):
    name = cid = 'T2-R3-05'
    in_kind, in_dim, out_dim, l_max = 'vec', 1, 49, 784
    batch_size = 64
    chance, ceiling = 1 / 49, None
    @property
    def path(self):
        return self.data_dir
    def prepare(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / 'train_x.npy').exists():
            return
        x = np.load(self.path / 'imgs.npz')['arr_0']
        y = np.load(self.path / 'labels.npz')['arr_0']
        assert x.shape[0] == len(y) and y.min() == 0 and y.max() == 48
        np.save(self.path / 'train_x.npy', x)
        np.save(self.path / 'train_y.npy', y)

    def _read(self):
        return (np.load(self.path / 'train_x.npy'), np.load(self.path / 'train_y.npy'))

    def load(self, device):
        from .real import split_idx
        x, y = self._read()
        tr, va = split_idx(len(y), 5000 / len(y), 1)
        x = x.reshape(len(x), self.l_max, self.in_dim).astype(np.float32) / 255
        x = x[:, np.random.default_rng(0).permutation(self.l_max)]
        mu = x[tr].mean((0, 1))
        sd = x[tr].std((0, 1))
        x = (x - mu) / sd
        self.xtr = torch.from_numpy(x[tr]).to(device)
        self.xva = torch.from_numpy(x[va]).to(device)
        self.ytr = torch.from_numpy(y[tr]).long().to(device)
        self.yva = torch.from_numpy(y[va]).long().to(device)
        self.stats = (mu.tolist(), sd.tolist())

    def baseline(self):
        x, y = self._read()
        tr, va = split_idx(len(y), 5000 / len(y), 1)
        return {'baseline': self.majority(y[tr], y[va]), 'chance': self.chance, 'ceiling': self.ceiling, 'n_train': len(tr), 'n_val': len(va), 'class_freq_val': np.bincount(y[va], minlength=10).tolist()}

    def describe(self):
        return dict(candidate=self.cid, permutation_seed=0, split_seed=1, n_val=5000, l_max=self.l_max, in_dim=self.in_dim)
