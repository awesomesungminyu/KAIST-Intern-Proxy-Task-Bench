"""ETT 학습/평가 스크립트.

사용 (proxy_tasks/ 에서):
    python -m ett.train --model linear --seed 0 --dataset ETTh1
    python -m ett.train --model transformer --smoke_test
학습 설정은 experiments/tier2_compare/tier2_compare.py 와 같다. test 는 --final_test 일 때만 평가한다.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from common.gpu_guard import require_cuda
from common.paths import task_data, task_output
from common.seed import set_seed
from .data import load
from .models import MODELS, NO_TRAINING


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=MODELS.names(), default="linear")
    p.add_argument("--dataset", choices=["ETTh1", "ETTm1"], default="ETTh1")
    p.add_argument("--data_dir", type=str, default=None, help="기본: $PROXY_DATA_ROOT/ett")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default=None, help="기본: $PROXY_OUTPUT_ROOT/ett/<dataset>-<model>-s<seed>")
    p.add_argument("--smoke_test", action="store_true", help="1 epoch, train 3 batch")
    p.add_argument("--final_test", action="store_true", help="최종 보고용으로만 test 를 1회 평가")
    return p.parse_args()


def configure_backends():
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    se = ae = 0.0
    elements = 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        p = model(x)
        if not torch.isfinite(p).all():
            raise FloatingPointError("Nonfinite evaluation output")
        e = (p - y).double()
        se += e.square().sum().item()
        ae += e.abs().sum().item()
        elements += y.numel()
    return {"mse": se / elements, "mae": ae / elements}


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
    require_cuda("[ett]")
    set_seed(args.seed)
    configure_backends()

    sets, meta = load(args.dataset, Path(args.data_dir) if args.data_dir else task_data("ett"))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = MODELS.build(args.model).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ett] dataset={args.dataset} model={args.model} params={n_params:,}", flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=i == 0, generator=generator if i == 0 else None,
                          num_workers=0, pin_memory=True) for i, ds in enumerate(sets)]

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = task_output("ett", f"{args.dataset}-{args.model}", args.seed)

    history = []
    t0 = time.time()
    if args.model in NO_TRAINING:
        val = evaluate(model, loaders[1])
        history.append({"epoch": 0, "val_mse": val["mse"], "val_mae": val["mae"]})
        best, best_epoch, stop = val, 0, "no_training"
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best = best_state = None
        best_epoch = stale = 0
        stop = "max_epochs"
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = n = 0
            for i, (x, y) in enumerate(loaders[0], 1):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                loss_sum += loss.item() * len(y)
                n += len(y)
                if args.smoke_test and i >= 3:
                    break
            val = evaluate(model, loaders[1])
            history.append({"epoch": epoch, "train_loss": loss_sum / n, "val_mse": val["mse"], "val_mae": val["mae"]})
            print(f"[ett] epoch={epoch} train_loss={loss_sum / n:.5f} val_mse={val['mse']:.5f}", flush=True)
            if best is None or val["mse"] < best["mse"]:
                best, best_epoch, stale = val, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                stop = "early_stopping"
                break
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "best_model.pt")

    result = {"args": vars(args), "data": meta, "n_params": n_params, "history": history,
              "best_epoch": best_epoch, "val_mse": best["mse"], "val_mae": best["mae"], "stop_reason": stop,
              "total_time_sec": time.time() - t0, "test": None}
    if args.final_test:
        result["test"] = evaluate(model, loaders[2])
    with open(out_dir / "history.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ett] done. best_epoch={best_epoch} val_mse={best['mse']:.5f} -> {out_dir}")


if __name__ == "__main__":
    main()
