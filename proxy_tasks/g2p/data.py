"""G2P tsv 파일 + vocab.json 로더 및 배치 collate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


def load_vocab(path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    with open(path) as f:
        v = json.load(f)
    return v["char"], v["phone"]


def load_split(path: Path) -> List[Tuple[str, List[str]]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, phones = line.rstrip("\n").split("\t")
            items.append((word, phones.split()))
    return items


class G2PDataset(Dataset):
    def __init__(self, entries: List[Tuple[str, List[str]]], char_vocab: dict, phone_vocab: dict):
        self.entries = entries
        self.char_vocab = char_vocab
        self.phone_vocab = phone_vocab

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        word, phones = self.entries[idx]
        src = [self.char_vocab[c] for c in word]
        tgt = [BOS_ID] + [self.phone_vocab[p] for p in phones] + [EOS_ID]
        return src, tgt


def collate_fn(batch):
    src_len = max(len(s) for s, _ in batch)
    tgt_len = max(len(t) for _, t in batch)
    src = torch.full((len(batch), src_len), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), tgt_len), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:].clone()
    tgt_out[tgt_out == PAD_ID] = -100
    return {"src": src, "tgt_in": tgt_in, "tgt_out": tgt_out}
