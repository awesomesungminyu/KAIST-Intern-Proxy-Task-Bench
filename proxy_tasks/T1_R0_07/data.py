"""RegBench용 torch Dataset. 파일 없이 즉석 생성.

train/test는 서로 다른 시드 스트림에서 뽑아 disjoint PFA를 보장한다
(엄밀한 교집합 배제는 아니지만, 상태공간이 커서 충돌 확률은 무시 가능한 수준).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from .pfa import (
    NUM_SPECIAL_TOKENS,
    PAD,
    PFAConfig,
    build_icl_sequence,
    dfa_to_pfa,
    sample_dfa,
    vocab_size_with_specials,
)


@dataclass
class RegBenchDataConfig:
    pfa_cfg: PFAConfig
    examples_per_seq: int = 20  # 프롬프트 하나에 들어가는 in-context 예시 개수
    max_total_len: int = 256
    num_train_pfas: Optional[int] = None  # None = 매 아이템마다 새 DFA (무한 풀)
    virtual_epoch_size: int = 5000  # 논문의 "iteration당 5000 시퀀스" 관례를 따름


class RegBenchDataset(Dataset):
    def __init__(self, cfg: RegBenchDataConfig, seed: int, split: str):
        assert split in ("train", "test")
        self.cfg = cfg
        # train과 test가 절대 같은 RNG 스트림을 공유하지 않도록 split별로 offset
        self.base_seed = seed * 2 + (0 if split == "train" else 1_000_000_007)
        self.split = split
        self._fixed_pfas = None
        if cfg.num_train_pfas is not None:
            pool_rng = random.Random(self.base_seed)
            self._fixed_pfas = [
                dfa_to_pfa(sample_dfa(cfg.pfa_cfg, pool_rng))
                for _ in range(cfg.num_train_pfas)
            ]

    def __len__(self) -> int:
        return self.cfg.virtual_epoch_size

    def __getitem__(self, idx: int) -> List[int]:
        # 아이템별 독립 RNG: multi-worker DataLoader에서도 재현 가능
        item_rng = random.Random(self.base_seed * 9973 + idx)
        if self._fixed_pfas is not None:
            pfa = item_rng.choice(self._fixed_pfas)
        else:
            pfa = dfa_to_pfa(sample_dfa(self.cfg.pfa_cfg, item_rng))
        seq = build_icl_sequence(
            pfa,
            self.cfg.pfa_cfg,
            item_rng,
            num_examples=self.cfg.examples_per_seq,
            max_total_len=self.cfg.max_total_len,
        )
        if not seq:
            seq = [NUM_SPECIAL_TOKENS]  # 극히 드문 빈 시퀀스 방지용 fallback
        return seq


def collate_fn(batch: List[List[int]]) -> dict:
    max_len = max(len(s) for s in batch)
    input_ids = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    for i, s in enumerate(batch):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    # next-token prediction: x = seq[:-1], y = seq[1:]
    x = input_ids[:, :-1]
    y = input_ids[:, 1:].clone()
    y[y == PAD] = -100  # cross_entropy ignore_index
    return {"input_ids": x, "labels": y}
