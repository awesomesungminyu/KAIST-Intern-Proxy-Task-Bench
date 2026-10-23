"""ZINC-12K 학습/평가 스크립트. 평가지표는 문헌 관례에 맞춰 MAE.

사용:
    python -m zinc.prepare_data          # 최초 1회 (Dropbox 다운로드, 시간 걸림)
    python -m zinc.train --seed 0 --model gine
    python -m zinc.train --smoke_test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from common.seed import set_seed
from .models import MODELS
from common.gpu_guard import require_cuda
from common.paths import task_data, task_output


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=MODELS.names(), default="gine")
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--heads", type=int, default=8)          # gps 전용
    p.add_argument("--attn_dropout", type=float, default=0.0)  # gps 전용
    p.add_argument("--ckpt_epoch", type=int, default=200)   # 중간 지점 best-val 체크포인트
    p.add_argument("--root", type=str, default=None, help="기본: $PROXY_DATA_ROOT/zinc")
    p.add_argument("--epochs", type=int, default=200)  # Dwivedi et al. 관례상 100~400 epoch
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default=None, help="기본: $PROXY_OUTPUT_ROOT/zinc/<model>-s<seed>")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--validation_only", action="store_true", help="구조 탐색용: test를 로드하거나 평가하지 않음")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_abs_err, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        total_abs_err += (pred - batch.y).abs().sum().item()
        n += batch.y.size(0)
    model.train()
    return total_abs_err / max(n, 1)


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.batch_size = 32

    set_seed(args.seed)
    device = require_cuda()
    use_amp = args.amp and device == "cuda"

    root = Path(args.root) if args.root else task_data("zinc")
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} 없음 — 먼저 `python -m zinc.prepare_data`를 실행하세요."
        )
    meta = json.load(open(meta_path))
    num_atom_types = meta["num_atom_types"]
    num_bond_types = meta["num_bond_types"]

    train_ds = ZINC(root=str(root), subset=True, split="train")
    val_ds = ZINC(root=str(root), subset=True, split="val")
    test_ds = None if args.validation_only else ZINC(root=str(root), subset=True, split="test")
    if args.smoke_test:
        train_ds = train_ds[:200]
        val_ds = val_ds[:50]
        if test_ds is not None:
            test_ds = test_ds[:50]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = None if test_ds is None else DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MODELS.build(args.model, num_atom_types=num_atom_types, num_bond_types=num_bond_types,
                         hidden_dim=args.hidden_dim, heads=args.heads,
                         attn_dropout=args.attn_dropout).to(device)
    n_params = model.num_params()
    print(f"[zinc] model={args.model} params={n_params:,} device={device} "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds) if test_ds is not None else 'not_loaded'}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.L1Loss()  # MAE — 문헌 평가지표와 동일

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = task_output("zinc", args.model, args.seed)
    history = []
    best_val = float("inf")
    best_val_early = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_mae)
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        # epoch<=ckpt_epoch 구간의 best-val 체크포인트를 별도 보관 → 같은 학습 경로에서
        # 200/500 결과를 모두 얻는다(ReduceLROnPlateau는 val 이력만 보므로 궤적 동일).
        if epoch <= args.ckpt_epoch and val_mae < best_val_early:
            best_val_early = val_mae
            torch.save(model.state_dict(), out_dir / f"best_model_e{args.ckpt_epoch}.pt")

        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "val_mae": val_mae,
            "lr": opt.param_groups[0]["lr"],
            "elapsed_sec": elapsed,
        }
        history.append(rec)
        if epoch % 10 == 0 or epoch == args.epochs or args.smoke_test:
            print(f"[zinc] epoch={epoch} train_loss={rec['train_loss']:.4f} "
                  f"val_mae={val_mae:.4f} lr={rec['lr']:.2e} elapsed={elapsed:.1f}s")

    # best checkpoint로 최종 test MAE 측정 (문헌 관례)
    test_mae = None
    if test_loader is not None:
        model.load_state_dict(torch.load(out_dir / "best_model.pt"))
        test_mae = evaluate(model, test_loader, device)
    total_time = time.time() - t0
    print(f"[zinc] best_val_mae={best_val:.4f} test_mae={test_mae} total_time={total_time:.1f}s")
    print(f"[peak_vram] {torch.cuda.max_memory_allocated()/1024**3:.2f} GB", flush=True)

    # 중간 지점(ckpt_epoch) best-val 체크포인트의 test MAE도 측정
    early_ckpt = out_dir / f"best_model_e{args.ckpt_epoch}.pt"
    test_mae_early, time_at_early = None, None
    if test_loader is not None and args.epochs > args.ckpt_epoch and early_ckpt.exists():
        model.load_state_dict(torch.load(early_ckpt))
        test_mae_early = evaluate(model, test_loader, device)
        time_at_early = next((r["elapsed_sec"] for r in history
                              if r["epoch"] == args.ckpt_epoch), None)
        print(f"[zinc] e{args.ckpt_epoch} test_mae={test_mae_early:.4f} "
              f"time_at_e{args.ckpt_epoch}={time_at_early:.1f}s")

    with open(out_dir / "history.json", "w") as f:
        json.dump(
            {"args": vars(args), "history": history, "test_mae": test_mae,
             "total_time_sec": total_time, "num_params": n_params,
             "best_val": best_val, "best_val_early": best_val_early,
             "ckpt_epoch": args.ckpt_epoch, "test_mae_early": test_mae_early,
             "time_at_ckpt_epoch_sec": time_at_early},
            f, indent=2,
        )
    print(f"[zinc] done -> {out_dir}")


if __name__ == "__main__":
    main()
