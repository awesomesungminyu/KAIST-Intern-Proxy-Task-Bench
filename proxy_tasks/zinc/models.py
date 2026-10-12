"""ZINC-12K 그래프 회귀 베이스라인: GCN(edge feature 무시) / GINE(edge feature 사용).

파라미터 예산은 문헌 관례(Dwivedi et al. 2020, ~500K)를 기준으로 hidden_dim을
잡았다. 실제 파라미터 수는 build_model()이 반환하는 model.num_params()로 확인.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool


class GCNRegressor(nn.Module):
    def __init__(self, num_atom_types: int, hidden_dim: int = 145, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.atom_emb = nn.Embedding(num_atom_types, hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.dropout = dropout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data) -> torch.Tensor:
        x = self.atom_emb(data.x.squeeze(-1))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, data.batch)
        return self.readout(g).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GINERegressor(nn.Module):
    """edge feature(bond type)를 실제로 사용하는 GIN 변형(GINE, Hu et al. 2020)."""

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        hidden_dim: int = 110,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.atom_emb = nn.Embedding(num_atom_types, hidden_dim)
        self.bond_embs = nn.ModuleList(
            [nn.Embedding(num_bond_types, hidden_dim) for _ in range(n_layers)]
        )
        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
                    ),
                    edge_dim=hidden_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.dropout = dropout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data) -> torch.Tensor:
        x = self.atom_emb(data.x.squeeze(-1))
        for conv, norm, bond_emb in zip(self.convs, self.norms, self.bond_embs):
            e = bond_emb(data.edge_attr)
            x = F.relu(norm(conv(x, data.edge_index, e)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, data.batch)
        return self.readout(g).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(name: str, num_atom_types: int, num_bond_types: int, hidden_dim: int | None = None) -> nn.Module:
    """hidden_dim을 None으로 두면 문헌 관례상의 기본값을 쓴다. 실제 파라미터 수는
    항상 model.num_params()로 직접 확인하고, ~500K 예산에 맞춰 --hidden_dim으로
    조정할 것 (이 파일의 계산은 근사치이며 검증되지 않음)."""
    if name == "gcn":
        return GCNRegressor(num_atom_types, hidden_dim=hidden_dim or 145)
    if name == "gine":
        return GINERegressor(num_atom_types, num_bond_types, hidden_dim=hidden_dim or 110)
    raise ValueError(f"unknown model: {name}")
