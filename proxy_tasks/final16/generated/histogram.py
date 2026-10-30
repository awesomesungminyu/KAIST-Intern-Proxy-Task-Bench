"""T1-R2-01: positional histogram, causal seq2seq form. Input x_1..x_L SEP, then L blank slots; the target at slot j is the number of times
x_j occurs in the whole input (capped at 32, classes 1..32 -> 0..31). (L, V) per batch from {16,32,64} x {8,16,32}."""
import numpy as np
import torch
from torch.nn import functional as F

from ..base import batched
from .common import SynthPart


class Histogram(SynthPart):
    name = "histogram"
    in_kind, in_dim, out_dim, readout, l_max = "tok", 34, 32, "all", 129
    SEP, BLANK = 32, 33
    Ls, Vs, CAP = (16, 32, 64), (8, 16, 32), 32
    batch_size, n_eval = 64, 500
    score_name = "slot_accuracy"
    chance, ceiling = 1 / 32, 1.0

    def make(self, g, B, L, V):
        x = torch.randint(0, V, (B, L), generator=g)
        cnt = (x[:, :, None] == x[:, None, :]).sum(-1).clamp(max=self.CAP)
        seq = torch.cat([x, torch.full((B, 1), self.SEP), torch.full((B, L), self.BLANK)], 1)
        return seq, cnt - 1

    def build_val(self):
        out = {}
        for L in self.Ls:
            for V in self.Vs:
                out[(L, V)] = self.make(self.cpu_gen(2147470000 + 100 * L + V), self.n_eval, L, V)
        return out

    def sample(self, g):
        L = self.Ls[int(torch.randint(0, 3, (1,), generator=g))]
        V = self.Vs[int(torch.randint(0, 3, (1,), generator=g))]
        x, y = self.make(g, self.batch_size, L, V)
        return {"x": x.to(self.device), "y": y.to(self.device), "L": L}

    def loss(self, out, batch):
        L = batch["L"]
        return F.cross_entropy(out[:, L + 1:].reshape(-1, self.out_dim), batch["y"].reshape(-1))

    def evaluate(self, model):
        accs = {}
        for (L, V), (x, y) in self.val.items():
            out = batched(model, lambda lo, hi: (x[lo:hi], None), x.shape[0], 250)[:, L + 1:]
            accs[(L, V)] = (out.argmax(-1) == y).float().mean().item()
        r = {"score": sum(accs.values()) / len(accs)}
        r.update({f"acc_L{L}_V{V}": a for (L, V), a in accs.items()})
        return r

    def baseline(self):
        ys = []
        for L in self.Ls:
            for V in self.Vs:
                ys.append(self.make(self.cpu_gen(777 + 100 * L + V), 5000, L, V)[1].reshape(-1).numpy())
        maj = np.bincount(np.concatenate(ys)).argmax()
        val = [self.make(self.cpu_gen(2147470000 + 100 * L + V), self.n_eval, L, V)[1].reshape(-1).numpy()
               for L in self.Ls for V in self.Vs]
        accs = [float((v == maj).mean()) for v in val]
        return {"baseline": float(np.mean(accs)), "chance": self.chance, "ceiling": self.ceiling,
                "majority_count_class": int(maj) + 1}

    def describe(self):
        return {"Ls": list(self.Ls), "Vs": list(self.Vs), "count_cap": self.CAP, "n_eval_per_combo": self.n_eval,
                "form": "causal seq2seq: inputs, SEP, L blank slots; slot j = count of x_j in the whole input"}
