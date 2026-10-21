"""sCIFAR 학습/평가 스크립트.

사용 (proxy_tasks/ 에서):
    python -m scifar.train --model transformer --seed 0
    python -m scifar.train --model lstm --smoke_test
학습 설정은 experiments/tier2_compare/tier2_compare.py 와 같다 (고정 epoch, validation 정확도 최고 checkpoint 선택).
test 는 --final_test 일 때만 평가한다.
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
from .data import load, normalize
from .models import MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=MODELS.names(), default="transformer")
    p.add_argument("--data_dir", type=str, default=None, help="기본: $PROXY_DATA_ROOT/scifar (그 아래 cifar/)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--out_dir", type=str, default=None, help="기본: $PROXY_OUTPUT_ROOT/scifar/<model>-s<seed>")
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
def evaluate(model, loader, stats):
    model.eval()
    n = correct = 0
    loss = 0.0
    for x, y in loader:
        x, y = normalize(x.cuda(non_blocking=True), stats), y.cuda(non_blocking=True)
        p = model(x)
        if not torch.isfinite(p).all():
            raise FloatingPointError("Nonfinite evaluation output")
        loss += F.cross_entropy(p, y, reduction="sum").item()
        correct += (p.argmax(-1) == y).sum().item()
        n += len(y)
    return {"accuracy": correct / n, "loss": loss / n}


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
    require_cuda("[scifar]")
    set_seed(args.seed)
    configure_backends()

    sets, meta, stats = load(Path(args.data_dir) if args.data_dir else task_data("scifar"))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    stats = tuple(torch.tensor(v, dtype=torch.float32, device="cuda").view(1, 3, 1) for v in stats)
    model = MODELS.build(args.model).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[scifar] model={args.model} params={n_params:,}", flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=i == 0, generator=generator if i == 0 else None,
                          num_workers=0, pin_memory=True) for i, ds in enumerate(sets)]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = task_output("scifar", args.model, args.seed)

    history = []
    best = best_state = None
    best_epoch = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = n = 0
        for i, (x, y) in enumerate(loaders[0], 1):
            x, y = normalize(x.cuda(non_blocking=True), stats), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
            if args.smoke_test and i >= 3:
                break
        val = evaluate(model, loaders[1], stats)
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "val_accuracy": val["accuracy"],
                        "val_loss": val["loss"]})
        print(f"[scifar] epoch={epoch} train_loss={loss_sum / n:.4f} val_acc={val['accuracy']:.4f}", flush=True)
        if best is None or val["accuracy"] > best["accuracy"]:
            best, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_model.pt")

    result = {"args": vars(args), "data": meta, "n_params": n_params, "history": history,
              "best_epoch": best_epoch, "val_accuracy": best["accuracy"], "val_loss": best["loss"],
              "total_time_sec": time.time() - t0, "test": None}
    if args.final_test:
        result["test"] = evaluate(model, loaders[2], stats)
    with open(out_dir / "history.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[scifar] done. best_epoch={best_epoch} val_acc={best['accuracy']:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
