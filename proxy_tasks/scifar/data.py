"""sequential CIFAR-10. 원본: experiments/tier2_compare/tier2_compare.py 의 data_for('sCIFAR').

train 50,000 장을 고정 seed 42 로 45,000 / 5,000 (train / validation) 로 나누고, 공식 test 10,000 장은 따로 둔다.
uint8 그대로 담고, 정규화는 GPU 에서 batch 마다 한다 (normalize()).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset


def load(data_dir: Path):
    from torchvision.datasets import CIFAR10

    root = str(Path(data_dir) / "cifar")
    train = CIFAR10(root, train=True, download=False)
    test = CIFAR10(root, train=False, download=False)
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(42))
    ti, vi = perm[:45000], perm[45000:]
    arr = np.ascontiguousarray(train.data.transpose(0, 3, 1, 2).reshape(-1, 3, 1024))
    assert np.array_equal(arr[123].reshape(3, 32, 32).transpose(1, 2, 0), train.data[123])
    sm = np.zeros(3)
    sq = np.zeros(3)
    n = 0
    for ix in ti.split(1000):
        z = arr[ix.numpy()].astype(np.float64) / 255
        sm += z.sum((0, 2))
        sq += (z * z).sum((0, 2))
        n += z.shape[0] * 1024
    mu = sm / n
    sd = np.sqrt(sq / n - mu ** 2)
    full = TensorDataset(torch.from_numpy(arr), torch.tensor(train.targets))
    tx = np.ascontiguousarray(test.data.transpose(0, 3, 1, 2).reshape(-1, 3, 1024))
    sets = [Subset(full, ti.tolist()), Subset(full, vi.tolist()),
            TensorDataset(torch.from_numpy(tx), torch.tensor(test.targets))]
    meta = dict(input_shape=[1024, 3], split_seed=42, sample_counts=[45000, 5000, 10000],
                channel_mean=mu.tolist(), channel_std=sd.tolist(), augmentation="none",
                normalization="training subset only")
    return sets, meta, (mu, sd)


def normalize(x, stats):
    """(B, 3, 1024) uint8 → (B, 1024, 3) float32."""
    return ((x.float() / 255 - stats[0]) / stats[1]).transpose(1, 2).contiguous()
