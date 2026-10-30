"""T1-R2-02: two-string match. `A SEP B`: is B == A? Negatives: 25% random mismatch + 25% exchange (two different-valued positions of A
swapped, so token frequencies match). Class balanced. Lengths n per batch from 8..64; evaluation n in {16, 32, 64}."""
import numpy as np
import torch
from torch.nn import functional as F

from .common import SynthPart


class StringMatch(SynthPart):
    name = "string_match"
    in_kind, in_dim, out_dim, readout, l_max = "tok", 17, 2, "last", 129
    V, SEP = 16, 16
    batch_size = 64
    n_train_range, eval_ns, n_eval = (8, 64), (16, 32, 64), 2000
    score_name = "accuracy"
    chance, ceiling = 0.5, 1.0

    def make(self, g, B, n):
        A = torch.randint(0, self.V, (B, n), generator=g)
        r = torch.rand(B, generator=g)
        pos, rnd = r < 0.5, (r >= 0.5) & (r < 0.75)
        Bm = A.clone()
        # exchange negatives: swap A[i] with a position j holding a different token
        i = torch.randint(0, n, (B,), generator=g)
        ar = torch.arange(B)
        diff = A != A[ar, i][:, None]
        j = (torch.rand(B, n, generator=g) * diff).argmax(1)
        ok = diff.any(1)
        swap = (~pos) & (~rnd) & ok
        Bm[ar[swap], i[swap]] = A[ar[swap], j[swap]]
        Bm[ar[swap], j[swap]] = A[ar[swap], i[swap]]
        rnd = rnd | ((~pos) & (~rnd) & ~ok)               # (practically never) no differing token: fall back to a random string
        R = torch.randint(0, self.V, (B, n), generator=g)
        same = (R == A).all(1)
        R[same, 0] = (R[same, 0] + 1) % self.V
        Bm[rnd] = R[rnd]
        x = torch.cat([A, torch.full((B, 1), self.SEP), Bm], 1)
        kind = torch.zeros(B, dtype=torch.long)           # 0 positive, 1 random mismatch, 2 exchange
        kind[rnd] = 1
        kind[swap] = 2
        return x, pos.long(), kind

    def build_val(self):
        self.kinds = {}
        out = {}
        for n in self.eval_ns:
            x, y, k = self.make(self.cpu_gen(2147480000 + n), self.n_eval, n)
            out[n] = (x, y)
            self.kinds[n] = k.to(self.device)
        return out

    def sample(self, g):
        n = int(torch.randint(self.n_train_range[0], self.n_train_range[1] + 1, (1,), generator=g))
        x, y, _ = self.make(g, self.batch_size, n)
        return {"x": x.to(self.device), "y": y.to(self.device)}

    def loss(self, out, batch):
        return F.cross_entropy(out, batch["y"])

    def evaluate(self, model):
        accs, swap = {}, {}
        for n, (x, y) in self.val.items():
            out = self.batched_pred(model, x)
            correct = (out == y)
            accs[n] = correct.float().mean().item()
            m = self.kinds[n] == 2
            swap[n] = correct[m | (self.kinds[n] == 0)].float().mean().item()      # positives vs exchange negatives only
        return {"score": sum(accs.values()) / len(accs), **{f"acc_n{n}": a for n, a in accs.items()},
                **{f"acc_pos_vs_exchange_n{n}": a for n, a in swap.items()}}

    def batched_pred(self, model, x):
        from ..base import batched
        return batched(model, lambda lo, hi: (x[lo:hi], None), x.shape[0], 250).argmax(-1)

    def baseline(self):
        ys = [self.make(self.cpu_gen(2147480000 + n), self.n_eval, n)[1] for n in self.eval_ns]
        y = torch.cat(ys).numpy()
        return {"baseline": float(max(y.mean(), 1 - y.mean())), "chance": self.chance, "ceiling": self.ceiling,
                "positive_fraction": float(y.mean())}

    def describe(self):
        return {"n_train_range": list(self.n_train_range), "eval_ns": list(self.eval_ns), "n_eval_per_n": self.n_eval,
                "negatives": "25% random mismatch + 25% exchange (same multiset), 50% positive"}
