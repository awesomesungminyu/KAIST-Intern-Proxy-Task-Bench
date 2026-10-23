"""Speech Commands 12-class 원본 파형 캐시.

`<data_dir>/{training,validation,testing}_{x,y}.npy` (x: float32 (N, 16000), y: int (N,)).
캐시는 experiments/scraw/scraw_bootstrap.py 가 기존 12-class 캐시(KWT 세팅의 unknown/silence 선택,
seed 456)를 원본 파형으로 다시 쓴 것이다. 원본: experiments/tier2_followup30/tier2_raw_followup.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPLIT_SIZES = {"training": 36923, "validation": 4443, "testing": 4888}


class RawDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, indices=None):
        self.x = np.load(Path(data_dir) / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(Path(data_dir) / f"{split}_y.npy", mmap_mode="r")
        if self.x.shape != (SPLIT_SIZES[split], 16000) or len(self.y) != SPLIT_SIZES[split]:
            raise ValueError(f"{data_dir}/{split}: unexpected shape {self.x.shape}, {len(self.y)}")
        self.indices = np.arange(len(self.y)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return torch.from_numpy(np.array(self.x[j], copy=True))[:, None], int(self.y[j])
