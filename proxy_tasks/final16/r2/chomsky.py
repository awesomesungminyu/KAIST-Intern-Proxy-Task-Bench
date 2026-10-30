import numpy as np
import torch
from torch.nn import functional as F
from .common import SynthPart

class LenExtrapPart(SynthPart):
    """Base for the 'ID + extrapolation' classification Parts: evaluation sets keyed ('id'|'ex', n)."""
    batch_size, n_eval = (64, 500)
    id_ns, ex_ns, train_range = ((), (), (1, 1))
    score_name = 'accuracy_id50_ex50'

    def build_val(self):
        return {(grp, n): self.make(self.cpu_gen(2147430000 + 7 * n + (grp == 'ex')), self.n_eval, n) for grp, ns in (('id', self.id_ns), ('ex', self.ex_ns)) for n in ns}

    def train_n(self, g):
        return int(torch.randint(self.train_range[0], self.train_range[1] + 1, (1,), generator=g))

    def sample(self, g):
        x, y = self.make(g, self.batch_size, self.train_n(g))
        return {'x': x.to(self.device), 'y': y.to(self.device)}

    def loss(self, out, batch):
        return F.cross_entropy(out, batch['y'])

    def metric(self, model, x, y, n):
        return self.acc(model, x, y)

    def evaluate(self, model):
        accs = {k: self.metric(model, x, y, k[1]) for k, (x, y) in self.val.items()}
        idv = float(np.mean([a for (g, _), a in accs.items() if g == 'id']))
        exv = float(np.mean([a for (g, _), a in accs.items() if g == 'ex']))
        r = {'score': 0.5 * idv + 0.5 * exv, 'acc_id': idv, 'acc_extrap': exv}
        r.update({f'acc_{g}_n{n}': a for (g, n), a in accs.items()})
        return r

    def describe(self):
        return {'id_ns': list(self.id_ns), 'extrap_ns': list(self.ex_ns), 'train_n_range': list(self.train_range), 'n_eval_per_n': self.n_eval, 'score': '0.5 * mean(ID accuracies) + 0.5 * mean(extrapolation accuracies)'}

class ModArith(LenExtrapPart):
    """Expressions over digits 0-4 with + - * (multiplication binds tighter), mod 5, written as tokens and ended by '='. Number of
    operands n: training 2..20; ID evaluation n in {5,10,15,20}; extrapolation n in {25,30,35,40}. Output: the value (5 classes)."""
    name = 'mod_arith_simple'
    in_kind, in_dim, out_dim, readout, l_max = ('tok', 9, 5, 'last', 80)
    EQ = 8
    id_ns, ex_ns, train_range = ((5, 10, 15, 20), (25, 30, 35, 40), (2, 20))
    chance, ceiling = (0.2, 1.0)

    def make(self, g, B, n):
        nums = torch.randint(0, 5, (B, n), generator=g)
        ops = torch.randint(0, 3, (B, n - 1), generator=g)
        x = torch.zeros(B, 2 * n, dtype=torch.long)
        x[:, 0:2 * n - 1:2] = nums
        x[:, 1:2 * n - 1:2] = ops + 5
        x[:, -1] = self.EQ
        S = torch.zeros(B, dtype=torch.long)
        term = nums[:, 0].clone()
        sign = torch.ones(B, dtype=torch.long)
        for t in range(n - 1):
            op, b = (ops[:, t], nums[:, t + 1])
            mul = op == 2
            S = torch.where(mul, S, (S + sign * term) % 5)
            sign = torch.where(mul, sign, torch.where(op == 0, torch.ones_like(sign), -torch.ones_like(sign)))
            term = torch.where(mul, term * b % 5, b)
        return (x, (S + sign * term) % 5)

    def baseline(self):
        return {'baseline': self.chance, 'chance': self.chance, 'ceiling': self.ceiling, 'note': 'results are uniform over 5 classes up to sampling noise; baseline = chance'}

class CycleNav(LenExtrapPart):
    """Moves {-1, 0, +1} (tokens 0,1,2) on a cycle of size 5 from position 0, then END (token 3); output the final position.
    n moves: training 1..40; ID evaluation n in {10,20,30,40}; extrapolation n in {50,60,80,100}."""
    name = 'cycle_navigation'
    in_kind, in_dim, out_dim, readout, l_max = ('tok', 4, 5, 'last', 101)
    END = 3
    id_ns, ex_ns, train_range = ((10, 20, 30, 40), (50, 60, 80, 100), (1, 40))
    chance, ceiling = (0.2, 1.0)

    def make(self, g, B, n):
        mv = torch.randint(0, 3, (B, n), generator=g)
        y = (mv - 1).sum(1) % 5
        return (torch.cat([mv, torch.full((B, 1), self.END)], 1), y)

    def baseline(self):
        return {'baseline': self.chance, 'chance': self.chance, 'ceiling': self.ceiling, 'note': 'final position is near-uniform over 5 classes; baseline = chance'}
