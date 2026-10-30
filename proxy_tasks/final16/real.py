import gzip
import math
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from .base import ArrayCls, Part, batched

def split_idx(n, frac, seed, y=None):
    """Fixed validation indices (stratified when y is given): returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    if y is None:
        p = rng.permutation(n)
        k = int(round(n * frac))
        return (np.sort(p[k:]), np.sort(p[:k]))
    val = []
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0])
        val.append(ids[:int(round(len(ids) * frac))])
    val = np.sort(np.concatenate(val))
    return (np.setdiff1d(np.arange(n), val), val)
AA = 'ACDEFGHIKLMNPQRSTVWY'

class Stability(Part):
    """TAPE stability (HF AI4Protein/TAPE_Stability: train 53,614 / valid 2,512 / test 12,851, test not used).
    Regression on the standardised stability score; score = Spearman rho on valid. Improvement is judged on the
    standardised MSE against the train-mean predictor."""
    name = 'tape_stability'
    in_kind, in_dim, out_dim, readout, l_max, pad_id = ('tok', 22, 1, 'mean', 64, 0)
    batch_size = 128
    score_name = 'spearman_valid'
    kind = 'reg'
    chance, ceiling = (0.0, 0.7)

    @property
    def path(self):
        return self.data_dir

    def _read(self, split):
        import pandas as pd
        df = pd.read_csv(self.path / f'{split}.csv')
        table = np.full(256, 21, np.uint8)
        for i, c in enumerate(AA):
            table[ord(c)] = i + 1
        x = np.zeros((len(df), self.l_max), np.uint8)
        for i, s in enumerate(df.aa_seq.values):
            b = table[np.frombuffer(s.encode('ascii'), np.uint8)[:self.l_max]]
            x[i, :len(b)] = b
        return (x, df.label.values.astype(np.float32), df.aa_seq.str.len().values)

    def load(self, device):
        (xtr, ytr, ltr), (xva, yva, lva) = (self._read('train'), self._read('valid'))
        assert max(ltr.max(), lva.max()) <= self.l_max
        self.mu, self.sd = (ytr.mean(), ytr.std())
        self.xtr, self.xva = (torch.from_numpy(xtr).to(device), torch.from_numpy(xva).to(device))
        self.ytr = torch.from_numpy((ytr - self.mu) / self.sd).to(device)
        self.yva = torch.from_numpy((yva - self.mu) / self.sd).to(device)

    def sample(self, g):
        i = torch.randint(self.xtr.shape[0], (self.batch_size,), generator=g, device=self.xtr.device)
        return {'x': self.xtr[i].long(), 'y': self.ytr[i]}

    def loss(self, out, batch):
        return F.mse_loss(out.squeeze(-1), batch['y'])

    def _eval(self, model, x, y):
        from scipy.stats import spearmanr
        out = batched(model, lambda lo, hi: (x[lo:hi].long(), x[lo:hi] != 0), x.shape[0], 512).squeeze(-1)
        rho = spearmanr(out.cpu().numpy(), y.cpu().numpy()).correlation
        return {'score': float(rho), 'mse_standardised': F.mse_loss(out, y).item()}

    def evaluate(self, model):
        r = self._eval(model, self.xva, self.yva)
        r['improve_value'] = r['mse_standardised']
        return r

    def evaluate_train(self, model):
        return self._eval(model, self.xtr[:2000], self.ytr[:2000])

    def baseline(self):
        (_, ytr, _), (_, yva, _) = (self._read('train'), self._read('valid'))
        mse = float((((yva - ytr.mean()) / ytr.std()) ** 2).mean())
        return {'baseline': 0.0, 'improve_baseline': mse, 'chance': 0.0, 'ceiling': self.ceiling, 'n_train': len(ytr), 'n_val': len(yva)}

class ImdbBytes(ArrayCls):
    """LRA-Text style byte-level IMDB (HF stanfordnlp/imdb, 25k train reviews): UTF-8 bytes + 1, truncated to 2,048,
    validation = 2,500 reviews (fixed seed 0) held out of the training split; the 25k test split is not used."""
    name = 'imdb_bytes'
    in_kind, in_dim, out_dim, l_max, pad_id = ('tok', 257, 2, 2048, 0)
    batch_size = 16
    trim = True
    chance, ceiling = (0.5, None)

    @property
    def path(self):
        return self.data_dir

    def prepare(self):
        if (self.path / 'val_y.npy').exists():
            return
        import pandas as pd
        df = pd.read_parquet(self.path / 'train.parquet')
        x = np.zeros((len(df), self.l_max), np.uint16)
        for i, s in enumerate(df.text.values):
            b = np.frombuffer(s.encode('utf-8')[:self.l_max], np.uint8).astype(np.uint16) + 1
            x[i, :len(b)] = b
        y = df.label.values.astype(np.uint8)
        tr, va = split_idx(len(y), 0.1, 0)
        for s, ids in (('train', tr), ('val', va)):
            np.save(self.path / f'{s}_x.npy', x[ids])
            np.save(self.path / f'{s}_y.npy', y[ids])

    def load(self, device):
        for k in ('x', 'y'):
            for s, n in (('train', 'tr'), ('val', 'va')):
                setattr(self, f'{k}{n}', torch.from_numpy(np.load(self.path / f'{s}_{k}.npy').astype(np.int64)).to(device))

    def baseline(self):
        ytr, yva = (np.load(self.path / 'train_y.npy'), np.load(self.path / 'val_y.npy'))
        return {'baseline': self.majority(ytr, yva), 'chance': self.chance, 'ceiling': None, 'n_train': len(ytr), 'n_val': len(yva)}

class Enwik8(Part):
    """enwik8 characters (bytes), context 256. Training windows are drawn from the first 90M bytes; validation is the
    first 1M bytes after them in non-overlapping 256-token windows, every position scored (bits per character)."""
    name = 'enwik8'
    in_kind, in_dim, out_dim, readout, l_max = ('tok', 256, 256, 'all', 256)
    ctx = 256
    batch_size = 64
    score_name = 'val_bpc'
    direction = 'min'
    kind = 'reg'
    chance, ceiling = (8.0, 1.5)

    @property
    def file(self):
        return self.data_dir / 'enwik8.zip'

    def _bytes(self):
        return np.frombuffer(zipfile.ZipFile(self.file).read('enwik8'), np.uint8)

    def load(self, device):
        raw = self._bytes()
        assert len(raw) == 100000000
        self.data = torch.from_numpy(raw.copy()).to(device)
        self.device = device
        self.ar = torch.arange(self.ctx + 1, device=device)
        self.val_starts = torch.arange(90000000, 91000000 - self.ctx - 1, self.ctx, device=device)

    def sample(self, g):
        s = torch.randint(0, 90000000 - self.ctx - 1, (self.batch_size,), generator=g, device=self.device)
        c = self.data[s[:, None] + self.ar[None]].long()
        return {'x': c[:, :-1], 'y': c[:, 1:]}

    def loss(self, out, batch):
        return F.cross_entropy(out.reshape(-1, 256), batch['y'].reshape(-1))

    @torch.no_grad()
    def _bpc(self, model, starts):
        tot = 0.0
        for lo in range(0, starts.shape[0], 256):
            c = self.data[starts[lo:lo + 256, None] + self.ar[None]].long()
            out = model(c[:, :-1])
            tot += F.cross_entropy(out.reshape(-1, 256), c[:, 1:].reshape(-1), reduction='sum').item()
        return tot / (starts.shape[0] * self.ctx) / math.log(2)

    def evaluate(self, model):
        return {'score': self._bpc(model, self.val_starts)}

    def evaluate_train(self, model):
        return {'score': self._bpc(model, self.val_starts - 90000000 + 1000000)}

    def baseline(self):
        raw = self._bytes()
        p = np.bincount(raw[:90000000], minlength=256) / 90000000
        val = raw[90000000:91000000]
        bits = float(-np.log2(np.maximum(p[val], 1e-12)).mean())
        return {'baseline': bits, 'chance': 8.0, 'ceiling': self.ceiling, 'unigram_entropy_train_bits': float(-(p[p > 0] * np.log2(p[p > 0])).sum())}

class PsMnist(ArrayCls):
    """Permuted sequential MNIST: 784 steps x 1 channel, fixed pixel permutation (seed 0), standardised with the MNIST
    mean/std; validation = 5,000 training images (fixed seed 1); the 10k test set is not used."""
    name = 'psmnist'
    in_kind, in_dim, out_dim, l_max = ('vec', 1, 10, 784)
    batch_size = 64
    chance, ceiling = (0.1, 0.9)

    @property
    def path(self):
        return self.data_dir

    def _read(self):

        def idx(name):
            with gzip.open(self.path / name) as f:
                buf = f.read()
            return np.frombuffer(buf, np.uint8, offset=16 if 'images' in name else 8)
        x = idx('train-images-idx3-ubyte.gz').reshape(-1, 784)
        y = idx('train-labels-idx1-ubyte.gz')
        return (x, y)

    def prepare(self):
        pass

    def load(self, device):
        x, y = self._read()
        perm = np.random.default_rng(0).permutation(784)
        x = ((x[:, perm].astype(np.float32) / 255.0 - 0.1307) / 0.3081)[..., None]
        tr, va = split_idx(len(y), 5000 / len(y), 1)
        self.xtr, self.ytr = (torch.from_numpy(x[tr]).to(device), torch.from_numpy(y[tr].astype(np.int64)).to(device))
        self.xva, self.yva = (torch.from_numpy(x[va]).to(device), torch.from_numpy(y[va].astype(np.int64)).to(device))

    def baseline(self):
        x, y = self._read()
        tr, va = split_idx(len(y), 5000 / len(y), 1)
        return {'baseline': self.majority(y[tr], y[va]), 'chance': self.chance, 'ceiling': self.ceiling, 'n_train': len(tr), 'n_val': len(va)}
