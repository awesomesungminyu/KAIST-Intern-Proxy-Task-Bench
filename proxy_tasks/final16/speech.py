"""35-class Speech Commands, raw waveform resampled from 16kHz to 2kHz."""
import numpy as np
import torch
from .base import ArrayCls

class SpeechCommands(ArrayCls):
    name = 'T2-R4-02'
    in_kind, in_dim, out_dim, l_max = 'vec', 1, 35, 2000
    batch_size, eval_bs = 32, 32
    ceiling = None
    @property
    def path(self):
        return self.data_dir
    def load(self, device):
        for split, suffix in [('train', 'tr'), ('val', 'va')]:
            for kind in ['x', 'y']:
                a = np.load(self.path / f'{split}_{kind}.npy')
                if a.dtype == np.uint16:
                    a = a.astype(np.int32)
                setattr(self, kind + suffix, torch.from_numpy(a).to(device))
    def describe(self):
        return dict(candidate=self.name, shape=[self.l_max, self.in_dim], n_train=len(self.ytr), n_val=len(self.yva))
