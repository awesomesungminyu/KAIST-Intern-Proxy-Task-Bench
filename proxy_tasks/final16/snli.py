"""Original round-2 SNLI bytes, splits, tokenization and metric; portable paths only."""
import json
from collections import Counter

import numpy as np
import torch

from .base import ArrayCls

SEP_BYTE = 0x1F                                   # ASCII unit separator between segments (token id 32)


def to_bytes(s, limit, keep="end"):
    b = s.encode("utf-8")
    return b[-limit:] if keep == "tail" else b[:limit]


class ByteTextCls(ArrayCls):
    in_kind, in_dim, readout, pad_id, trim = "tok", 257, "mean", 0, True
    folder = None
    chance, ceiling = None, None

    @property
    def path(self):
        return self.data_dir

    @property
    def l_max(self):
        return int(json.loads((self.path / "meta.json").read_text())["max_len"])

    def _save(self, name, rows, labels, L, extra=None):
        x = np.zeros((len(rows), L), np.uint16)
        for i, b in enumerate(rows):
            b = np.frombuffer(b[:L], np.uint8).astype(np.uint16) + 1
            x[i, :len(b)] = b
        np.save(self.path / f"{name}_x.npy", x)
        np.save(self.path / f"{name}_y.npy", np.asarray(labels, np.int64))

    def load(self, device):
        for k in ("x", "y"):
            for s, n in (("train", "tr"), ("val", "va")):
                setattr(self, f"{k}{n}", torch.from_numpy(np.load(self.path / f"{s}_{k}.npy").astype(np.int64)).to(device))

    def baseline(self):
        ytr, yva = np.load(self.path / "train_y.npy"), np.load(self.path / "val_y.npy")
        return {"baseline": self.majority(ytr, yva), "chance": 1.0 / self.out_dim, "ceiling": self.ceiling,
                "n_train": int(len(ytr)), "n_val": int(len(yva)), "class_freq_val": np.bincount(yva, minlength=self.out_dim).tolist()}


# ------------------------------------------------------------------------------------------------ T2-R2-01
class SnliBytes(ByteTextCls):
    """premise + SEP + hypothesis, at most 256 bytes: when too long the premise is cut from its beginning so the hypothesis is always
    complete (a hypothesis longer than 255 bytes is cut at its end). 3 classes (entailment, neutral, contradiction); label -1 removed;
    official train / validation. The hypothesis-only naive Bayes accuracy is stored in baseline() as a REFERENCE only."""
    name = "snli_bytes"
    out_dim, batch_size = 3, 64
    folder, L = "t2x_snli", 256

    def prepare(self):
        if (self.path / "val_y.npy").exists():
            return
        import pyarrow.parquet as pq
        hyp = {}
        for split, name in (("train", "train"), ("validation", "val")):
            t = pq.read_table(self.path / f"{split}.parquet").to_pydict()
            rows, ys, hs = [], [], []
            for p, h, y in zip(t["premise"], t["hypothesis"], t["label"]):
                if y < 0:
                    continue
                hb = to_bytes(h, self.L - 1)
                pb = to_bytes(p, self.L - 1 - len(hb), keep="tail") if len(hb) < self.L - 1 else b""
                rows.append(pb + bytes([SEP_BYTE]) + hb)
                ys.append(y); hs.append(h)
            self._save(name, rows, ys, self.L)
            hyp[name] = (hs, ys)
        # reference: hypothesis-only multinomial naive Bayes over word unigrams (never used in any decision)
        cnt = [Counter(), Counter(), Counter()]
        for h, y in zip(*hyp["train"]):
            cnt[y].update(h.lower().split())
        vocab = {w for c in cnt for w, _ in c.most_common(20000)}
        tot = [sum(c[w] for w in vocab) + len(vocab) for c in cnt]
        prior = np.bincount(hyp["train"][1]) / len(hyp["train"][1])
        ok = 0
        for h, y in zip(*hyp["val"]):
            sc = [np.log(prior[k]) + sum(np.log((cnt[k][w] + 1) / tot[k]) for w in h.lower().split() if w in vocab) for k in range(3)]
            ok += int(np.argmax(sc) == y)
        (self.path / "meta.json").write_text(json.dumps({"max_len": self.L, "hypothesis_only_nb_val_accuracy": ok / len(hyp["val"][1])}))

    def baseline(self):
        d = super().baseline()
        d["hypothesis_only_nb_val_accuracy_REFERENCE_ONLY"] = json.loads((self.path / "meta.json").read_text())["hypothesis_only_nb_val_accuracy"]
        return d

    def describe(self):
        return {"max_len": self.L, "separator_byte": SEP_BYTE, "splits": "official train / validation, label -1 removed"}


