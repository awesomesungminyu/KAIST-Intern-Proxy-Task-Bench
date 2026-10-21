"""ETT(ETTh1 / ETTm1) 다변량 예측 데이터. 96 시점 입력 → 다음 96 시점 7채널.

원본: experiments/tier2_compare/tier2_compare.py 의 data_for(). 분할 경계·정규화·창 크기 동일.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
INPUT_STEPS = PREDICTION_STEPS = 96
# 원 실행(2026-10-16)이 기록한 CSV 해시. 상류 ETDataset 커밋은 experiments/quick_suite/quick_prepare.py.
SHA256 = {
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTm1": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
}


class Windows(Dataset):
    def __init__(self, x, start, end):
        self.x = x
        self.start = max(INPUT_STEPS, start)
        self.end = end

    def __len__(self):
        return self.end - self.start - INPUT_STEPS + 1

    def __getitem__(self, i):
        t = self.start + i
        return self.x[t - INPUT_STEPS:t], self.x[t:t + PREDICTION_STEPS]


def load(dataset: str, data_dir: Path):
    """(train, val, test) Windows 와 메타데이터. 정규화는 train 구간으로만 맞춘다."""
    path = Path(data_dir) / f"{dataset}.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[dataset]:
        raise ValueError(f"{path}: sha256 {digest} != expected {SHA256[dataset]}")
    df = pd.read_csv(path)
    a = df[CHANNELS].to_numpy(np.float64)
    assert list(df.columns) == ["date"] + CHANNELS and np.isfinite(a).all()
    f = 1 if dataset == "ETTh1" else 4
    b = [8640 * f, 11520 * f, 14400 * f]
    mu = a[:b[0]].mean(0)
    sd = a[:b[0]].std(0)
    assert (sd > 0).all()
    a = (a - mu) / sd
    x = torch.tensor(a, dtype=torch.float32)
    sets = [Windows(x, INPUT_STEPS, b[0]), Windows(x, b[0], b[1]), Windows(x, b[1], b[2])]
    assert [len(s) for s in sets] == ([8449, 2785, 2785] if f == 1 else [34369, 11425, 11425])
    meta = dict(dataset=dataset, data_sha256=digest, split_boundaries=b, channels=CHANNELS,
                normalization="train-only ddof=0 standardization", sample_counts=[len(s) for s in sets])
    return sets, meta
