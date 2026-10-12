"""ZINC-12K 다운로드 + 전처리 (PyTorch Geometric 표준 로더 사용).

주의: PyG의 ZINC(subset=True)는 Dropbox의 전체 250K molecules.zip을 받은 뒤
인덱스 파일로 12K만 필터링한다 (README의 "ZINC 주의사항" 참고). 이 sandbox
환경에서는 Dropbox가 방화벽에 막혀 있었지만, 일반적인 서버(예: kcvl-gpu3)에서는
보통 문제없이 받아진다.

atom_type / bond_type의 실제 vocab 크기를 데이터에서 직접 읽어 metadata.json에
저장한다 (문헌값을 하드코딩하지 않고 실측치를 씀 — "정확하게" 원칙).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch_geometric.datasets import ZINC

DEFAULT_ROOT = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    max_atom_type, max_bond_type = -1, -1
    stats = {}
    for split in ["train", "val", "test"]:
        ds = ZINC(root=str(root), subset=True, split=split)
        n_nodes = sum(d.num_nodes for d in ds)
        n_edges = sum(d.edge_index.size(1) for d in ds)
        max_atom_type = max(max_atom_type, max(int(d.x.max()) for d in ds))
        max_bond_type = max(max_bond_type, max(int(d.edge_attr.max()) for d in ds) if len(ds) else 0)
        stats[split] = {
            "num_graphs": len(ds),
            "avg_nodes": n_nodes / len(ds),
            "avg_edges": n_edges / len(ds),
        }
        print(f"[zinc] {split}: {len(ds)} graphs, avg_nodes={stats[split]['avg_nodes']:.2f}, "
              f"avg_edges={stats[split]['avg_edges']:.2f}")

    # 참고값(Dwivedi et al. 2020, Benchmarking GNNs): train 10000 / val 1000 / test 1000,
    # 평균 노드 약 23.15개. 위 print 결과와 이 값이 크게 다르면 다운로드/필터링이
    # 잘못됐다는 신호이니 반드시 대조해볼 것.
    meta = {
        "num_atom_types": max_atom_type + 1,
        "num_bond_types": max_bond_type + 1,
        "stats": stats,
    }
    with open(root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[zinc] num_atom_types={meta['num_atom_types']} num_bond_types={meta['num_bond_types']}")
    print(f"[zinc] wrote metadata -> {root / 'metadata.json'}")


if __name__ == "__main__":
    main()
