import numpy as np
import torch
from torch.nn import functional as F
from ..base import Part
from .common import R2_DATA

class Forecast(Part):
    in_kind, readout = ('vec', 'last')
    batch_size = 64
    direction, kind = ('min', 'reg')
    score_name = 'mse_ctx_id50_long50'
    chance, ceiling = (None, None)
    H, ctx_train, ctx_eval = (96, 96, (96, 336))

    @property
    def in_dim(self):
        return self._series()[0].shape[1]

    @property
    def out_dim(self):
        return self.H * len(self._series()[1])

    @property
    def l_max(self):
        return max(self.ctx_eval)

    def load(self, device):
        X, cols, ok, tr_end, va_start, va_end = self._series()
        self.dev = device
        self.X = torch.from_numpy(X).to(device)
        self.cols = cols
        okc = np.concatenate([[0], np.cumsum(~ok)])

        def valid(t0):
            return okc[t0 + self.H] - okc[t0] == 0
        lo = self.ctx_train
        self.t_train = torch.tensor([t for t in range(lo, tr_end - self.H + 1) if valid(t)], device=device)
        self.t_val = torch.tensor([t for t in range(va_start, va_end - self.H + 1) if valid(t)], device=device)
        assert int(self.t_val.min()) >= max(self.ctx_eval), 'validation windows need max(ctx_eval) steps of earlier history'

    def _window(self, t0, C):
        idx = t0[:, None] + torch.arange(-C, 0, device=t0.device)[None]
        return self.X[idx]

    def _target(self, t0):
        idx = t0[:, None] + torch.arange(0, self.H, device=t0.device)[None]
        return self.X[idx][:, :, self.cols].reshape(len(t0), -1)

    def sample(self, g):
        i = torch.randint(len(self.t_train), (self.batch_size,), generator=g, device=self.t_train.device)
        t0 = self.t_train[i]
        return {'x': self._window(t0, self.ctx_train), 'y': self._target(t0)}

    def loss(self, out, batch):
        return F.mse_loss(out, batch['y'])

    def _mse(self, model, C, ts=None):
        ts = self.t_val if ts is None else ts
        se, n = (0.0, 0)
        with torch.no_grad():
            for lo in range(0, len(ts), 256):
                t0 = ts[lo:lo + 256]
                out = model(self._window(t0, C), None)
                se += ((out - self._target(t0)) ** 2).sum().item()
                n += out.numel()
        return se / n

    def evaluate(self, model):
        m = {C: self._mse(model, C) for C in self.ctx_eval}
        return {'score': sum(m.values()) / len(m), **{f'mse_ctx{C}': v for C, v in m.items()}}

    def evaluate_train(self, model):
        ts = self.t_train[:1000]
        m = {C: self._mse(model, C, ts) for C in (self.ctx_train,)}
        return {'score': float(np.mean(list(m.values())))}

    def persistence(self):
        self.load('cpu')
        t0 = self.t_val
        last = self._window(t0, 1)[:, 0][:, self.cols]
        y = self._target(t0).reshape(len(t0), self.H, len(self.cols))
        return float(((y - last[:, None, :]) ** 2).mean())

    def baseline(self):
        p = self.persistence()
        return {'baseline': p, 'chance': p, 'ceiling': self.ceiling, 'n_train_windows': int(len(self.t_train)), 'n_val_windows': int(len(self.t_val)), 'note': 'baseline = persistence MSE on the validation windows (same for both contexts)'}

class BeijingPM25(Forecast):
    """UCI Beijing PM2.5 (hourly, 2010-2014, 43,824 rows). Train = 2010-2013, validation = 2014. Features: pm2.5, DEWP, TEMP, PRES, Iws, Is,
    Ir and a one-hot of the wind direction cbwd (11 inputs), standardised with the training statistics (pm2.5 statistics from its observed
    values). Missing pm2.5 is filled with the previous observation (the first ones with the training mean) and every window whose
    24 target hours contain an originally missing pm2.5 is skipped in training and evaluation. Horizon 24 h of pm2.5; contexts 48
    (training, ID) and 192 (long)."""
    name = 'beijing_pm25'
    H, ctx_train, ctx_eval = (24, 48, (48, 192))

    def _series(self):
        import pandas as pd
        df = pd.read_csv(R2_DATA / 'n10_beijing/x/PRSA_data_2010.1.1-2014.12.31.csv', na_values=['NA'])
        n_tr = int((df.year <= 2013).sum())
        pm = df['pm2.5'].values.astype(np.float64)
        ok = ~np.isnan(pm)
        mu_pm, sd_pm = (np.nanmean(pm[:n_tr]), np.nanstd(pm[:n_tr]))
        filled = pd.Series(pm).ffill().values.copy()
        filled[np.isnan(filled)] = mu_pm
        cols = [filled, df.DEWP.values, df.TEMP.values, df.PRES.values, df.Iws.values, df.Is.values, df.Ir.values]
        X = np.stack(cols, 1).astype(np.float64)
        mu, sd = (X[:n_tr].mean(0), X[:n_tr].std(0))
        mu[0], sd[0] = (mu_pm, sd_pm)
        X = (X - mu) / sd
        onehot = np.stack([df.cbwd.values == c for c in ('NE', 'NW', 'SE', 'cv')], 1).astype(np.float64)
        X = np.concatenate([X, onehot], 1).astype(np.float32)
        return (X, [0], ok, n_tr, n_tr, len(df))

    def describe(self):
        return {'train': '2010-2013', 'val': '2014', 'horizon': self.H, 'contexts': [self.ctx_train, *self.ctx_eval], 'missing': 'pm2.5 forward-filled; windows with a missing target hour skipped'}
