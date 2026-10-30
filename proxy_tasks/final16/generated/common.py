from ..base import batched
from ..compression import GenPart

class SynthPart(GenPart):
    """Generated data: training batches come from the CPU generator, evaluation sets are fixed (own seeds).
    Subclasses implement make(g, B, **cfg) -> (x, y) CPU tensors and list their evaluation sets in self.eval_sets()."""

    def to_dev(self, *ts):
        return tuple((t.to(self.device) for t in ts))

    def load(self, device):
        self.device = device
        self.val = {k: self.to_dev(*v) for k, v in self.build_val().items()}

    def build_val(self):
        raise NotImplementedError

    def acc(self, model, x, y):
        out = batched(model, lambda lo, hi: (x[lo:hi], None), x.shape[0], self.eval_bs)
        return (out.argmax(-1) == y).float().mean().item()
