"""ZINC-12K GraphGPS 회귀 모델 (GNN + Transformer 하이브리드, 순수 Transformer 아님).

각 레이어 = local message passing(GINEConv, 기존 GINE 베이스라인과 동일한 MLP 구성)
          + global multi-head self-attention.
Rampasek et al., "Recipe for a General, Powerful, Scalable Graph Transformer",
NeurIPS 2022 (GraphGPS).

Positional/structural encoding은 기본 구성에서 쓰지 않는다 — attention 추가 효과와
인코딩 추가 효과를 분리하기 위함(필요 시 별도 실험).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool


class GPSRegressor(nn.Module):
    def __init__(self, num_atom_types: int, num_bond_types: int, hidden_dim: int = 64,
                 n_layers: int = 4, heads: int = 8, dropout: float = 0.0,
                 attn_dropout: float = 0.0):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim({hidden_dim}) % heads({heads}) != 0")
        self.atom_emb = nn.Embedding(num_atom_types, hidden_dim)
        self.bond_embs = nn.ModuleList(
            [nn.Embedding(num_bond_types, hidden_dim) for _ in range(n_layers)])
        self.convs = nn.ModuleList([
            GPSConv(
                channels=hidden_dim,
                conv=GINEConv(
                    nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim)),
                    edge_dim=hidden_dim),
                heads=heads, dropout=dropout,
                attn_kwargs={"dropout": attn_dropout}, norm="batch_norm",
            ) for _ in range(n_layers)])
        # GPSConv가 내부 norm을 포함하므로 외부 BatchNorm은 두지 않는다.
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data) -> torch.Tensor:
        x = self.atom_emb(data.x.squeeze(-1))
        for conv, bond_emb in zip(self.convs, self.bond_embs):
            e = bond_emb(data.edge_attr)
            x = conv(x, data.edge_index, batch=data.batch, edge_attr=e)
        g = global_mean_pool(x, data.batch)
        return self.readout(g).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_gps(num_atom_types: int, num_bond_types: int, hidden_dim: int | None = None,
              n_layers: int = 4, heads: int = 8, dropout: float = 0.0,
              attn_dropout: float = 0.0) -> nn.Module:
    return GPSRegressor(num_atom_types, num_bond_types, hidden_dim=hidden_dim or 64,
                        n_layers=n_layers, heads=heads, dropout=dropout,
                        attn_dropout=attn_dropout)
