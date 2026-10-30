"""T1-R2-05: simulate a small random automaton. 6 states, alphabet of 3 letters, start state 0, output = the current state after every
letter. The transition table is the first one (seed 0, 1, 2, ...) that is strongly connected, has at least one letter that is not a
permutation of the states (so it is not a group automaton like T1-R0-04 / Tomita / S5) and for which fewer than 50% of the length-8
words determine the final state independently of the start state. Score = 0.5 * accuracy at the last position of length-32 inputs
+ 0.5 * mean accuracy at the last position of length-64 and length-96 inputs. Training lengths 1..32."""
import itertools

import numpy as np
import torch
from torch.nn import functional as F

from .chomsky import LenExtrapPart

Q, SIG = 6, 3


def _dfa(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, Q, size=(SIG, Q))                      # T[a][q]


def _strongly_connected(T):
    adj = [set(T[:, q].tolist()) for q in range(Q)]
    for s in range(Q):
        seen, stack = {s}, [s]
        while stack:
            for t in adj[stack.pop()]:
                if t not in seen:
                    seen.add(t); stack.append(t)
        if len(seen) < Q:
            return False
    return True


def _sync_fraction(T, k=8):
    ok = 0
    for w in itertools.product(range(SIG), repeat=k):
        s = np.arange(Q)
        for a in w:
            s = T[a][s]
        ok += len(set(s.tolist())) == 1
    return ok / SIG ** k


def find_dfa():
    for seed in range(1000):
        T = _dfa(seed)
        perm = [len(set(T[a].tolist())) == Q for a in range(SIG)]
        if _strongly_connected(T) and not all(perm) and _sync_fraction(T) < 0.5:
            return seed, T, _sync_fraction(T)
    raise RuntimeError("no automaton found")


class RandomAutomaton(LenExtrapPart):
    name = "random_automaton"
    in_kind, in_dim, out_dim, readout, l_max = "tok", SIG, Q, "all", 96
    id_ns, ex_ns, train_range = (32,), (64, 96), (1, 32)
    chance, ceiling = 1 / Q, 1.0
    n_eval = 2000
    score_name = "last_state_accuracy_id50_ex50"

    def __init__(self):
        self.dfa_seed, T, self.sync = find_dfa()
        self.T = torch.from_numpy(T)

    def make(self, g, B, n):
        x = torch.randint(0, SIG, (B, n), generator=g)
        s = torch.zeros(B, dtype=torch.long)
        ys = []
        for t in range(n):
            s = self.T[x[:, t], s]
            ys.append(s)
        return x, torch.stack(ys, 1)

    def loss(self, out, batch):
        return F.cross_entropy(out.reshape(-1, Q), batch["y"].reshape(-1))

    def metric(self, model, x, y, n):
        from ..base import batched
        out = batched(model, lambda lo, hi: (x[lo:hi], None), x.shape[0], 250)
        return (out[:, -1].argmax(-1) == y[:, -1]).float().mean().item()

    def baseline(self):
        ytr = np.concatenate([self.make(self.cpu_gen(777 + n), 5000, n)[1][:, -1].numpy() for n in (8, 16, 24, 32)])
        maj = np.bincount(ytr, minlength=Q).argmax()
        accs = {}
        for n in self.id_ns + self.ex_ns:
            accs[n] = float((self.make(self.cpu_gen(2147430000 + 7 * n + (n in self.ex_ns)), self.n_eval, n)[1][:, -1].numpy() == maj).mean())
        return {"baseline": 0.5 * accs[32] + 0.5 * np.mean([accs[64], accs[96]]), "chance": self.chance, "ceiling": self.ceiling,
                "majority_state": int(maj)}

    def describe(self):
        d = super().describe()
        d.update(dfa_seed=self.dfa_seed, transition_table=self.T.tolist(), sync_fraction_len8=self.sync)
        return d
