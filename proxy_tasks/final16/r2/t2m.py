"""T2-R2-02: SMILES Lipophilicity regression (MoleculeNet / DeepChem Lipophilicity.csv, 4,200 molecules). Bemis-Murcko scaffold split 80/10/10
with the DeepChem rule (scaffold groups sorted by size, largest first, filled into train until 80%, then validation until 90%, rest test;
chirality ignored); the 10% test part is not used. SMILES characters as tokens (vocabulary = characters of the whole file, 0 = PAD).
Label standardised with the training statistics, MSE training; score = validation RMSE in the original units (lower is better);
baseline = RMSE of the training-mean predictor."""
import numpy as np
import torch
from torch.nn import functional as F

from ..base import Part, batched, use_pylib
from .common import R2_DATA


def scaffold_split(smiles, frac_train=0.8, frac_valid=0.1):
    use_pylib()
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    groups = {}
    for i, s in enumerate(smiles):
        key = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False)
        groups.setdefault(key, []).append(i)
    sets = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(smiles)
    tr_cut, va_cut = frac_train * n, (frac_train + frac_valid) * n
    tr, va, te = [], [], []
    for g in sets:
        if len(tr) + len(g) > tr_cut:
            (te if len(tr) + len(va) + len(g) > va_cut else va).extend(g)
        else:
            tr.extend(g)
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


class Lipophilicity(Part):
    name = "lipophilicity"
    in_kind, out_dim, readout, pad_id = "tok", 1, "mean", 0
    batch_size = 64
    score_name = "rmse_valid"
    direction, kind = "min", "reg"
    chance, ceiling = None, None

    @property
    def path(self):
        return R2_DATA / "t2m_lipo"

    def _table(self):
        import pandas as pd
        df = pd.read_csv(self.path / "Lipophilicity.csv")
        smiles, y = df.smiles.tolist(), df.exp.values.astype(np.float32)
        chars = sorted({c for s in smiles for c in s})
        return smiles, y, {c: i + 1 for i, c in enumerate(chars)}

    @property
    def in_dim(self):
        return len(self._table()[2]) + 1

    @property
    def l_max(self):
        return max(len(s) for s in self._table()[0])

    def prepare(self):
        if (self.path / "split.npz").exists():
            return
        smiles, _, _ = self._table()
        tr, va, te = scaffold_split(smiles)
        np.savez(self.path / "split.npz", train=tr, val=va, test=te)

    def _arrays(self):
        smiles, y, vocab = self._table()
        sp = np.load(self.path / "split.npz")
        L = self.l_max
        x = np.zeros((len(smiles), L), np.int64)
        for i, s in enumerate(smiles):
            x[i, :len(s)] = [vocab[c] for c in s]
        return x, y, sp["train"], sp["val"]

    def load(self, device):
        x, y, tr, va = self._arrays()
        self.mu, self.sd = float(y[tr].mean()), float(y[tr].std())
        self.xtr, self.xva = torch.from_numpy(x[tr]).to(device), torch.from_numpy(x[va]).to(device)
        self.ytr = torch.from_numpy((y[tr] - self.mu) / self.sd).to(device)
        self.yva_raw = torch.from_numpy(y[va]).to(device)

    def sample(self, g):
        i = torch.randint(self.xtr.shape[0], (self.batch_size,), generator=g, device=self.xtr.device)
        return {"x": self.xtr[i], "y": self.ytr[i]}

    def loss(self, out, batch):
        return F.mse_loss(out.squeeze(-1), batch["y"])

    def _rmse(self, model, x, y_raw):
        out = batched(model, lambda lo, hi: (x[lo:hi], x[lo:hi] != 0), x.shape[0], 512).squeeze(-1) * self.sd + self.mu
        return ((out - y_raw) ** 2).mean().sqrt().item()

    def evaluate(self, model):
        return {"score": self._rmse(model, self.xva, self.yva_raw)}

    def evaluate_train(self, model):
        raw = self.ytr[:1000] * self.sd + self.mu
        return {"score": self._rmse(model, self.xtr[:1000], raw)}

    def baseline(self):
        x, y, tr, va = self._arrays()
        rmse = float(np.sqrt(((y[va] - y[tr].mean()) ** 2).mean()))
        return {"baseline": rmse, "chance": rmse, "ceiling": self.ceiling, "n_train": int(len(tr)), "n_val": int(len(va)),
                "n_test_unused": int(4200 - len(tr) - len(va)), "label_std_val": float(y[va].std())}

    def describe(self):
        return {"split": "Bemis-Murcko scaffold 80/10/10 (DeepChem rule)", "vocab_size": self.in_dim - 1, "max_len": self.l_max}
