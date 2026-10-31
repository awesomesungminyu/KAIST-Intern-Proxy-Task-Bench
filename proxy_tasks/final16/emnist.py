"""Round-3 EMNIST letters: identical split, orientation and permutation to the original."""
import gzip
import numpy as np
from .kuzushiji import Kuzushiji49


class EmnistLetters(Kuzushiji49):
    name = cid = 'T2-R3-02'
    out_dim = 26
    chance = 1 / 26

    def prepare(self):
        if (self.path / 'train_x.npy').exists() and (self.path / 'train_y.npy').exists():
            return
        with gzip.open(self.path / 'emnist-letters-train-images-idx3-ubyte.gz', 'rb') as stream:
            x = np.frombuffer(stream.read(), np.uint8, offset=16).reshape(-1, 28, 28).transpose(0, 2, 1)
        with gzip.open(self.path / 'emnist-letters-train-labels-idx1-ubyte.gz', 'rb') as stream:
            y = np.frombuffer(stream.read(), np.uint8, offset=8).astype(np.int64) - 1
        if len(x) != len(y) or y.min() != 0 or y.max() != 25:
            raise ValueError('Expected the official 26-class EMNIST letters training split')
        np.save(self.path / 'train_x.npy', x)
        np.save(self.path / 'train_y.npy', y)
