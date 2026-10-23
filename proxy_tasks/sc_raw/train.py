"""sc_raw 학습/평가 스크립트.

사용 (proxy_tasks/ 에서):
    python -m sc_raw.train --model lstm --seed 0
    python -m sc_raw.train --model transformer --smoke_test
학습 설정은 experiments/tier2_followup30/tier2_raw_followup.py 와 같다: train split 을 seed 로 섞은 앞
2×steps 개를 batch 2 로 한 번씩 보고(고정 update 예산), 최종 가중치로 validation 을 평가한다.
원 실행은 seed 0 만 썼다. test 는 --final_test 일 때만 평가한다.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from common.gpu_guard import require_cuda
from common.paths import task_data, task_output
from common.seed import set_seed
from .data import SPLIT_SIZES, RawDataset
from .models import MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=MODELS.names(), default="lstm")
    p.add_argument("--data_dir", type=str, default=None, help="기본: $PROXY_DATA_ROOT/sc_raw")
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--out_dir", type=str, default=None, help="기본: $PROXY_OUTPUT_ROOT/sc_raw/<model>-s<seed>")
    p.add_argument("--smoke_test", action="store_true", help="3 update")
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
    n = correct = 0
    loss = 0.0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
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
        args.steps = 3
    require_cuda("[sc_raw]")
    set_seed(args.seed)
    configure_backends()
    data_dir = Path(args.data_dir) if args.data_dir else task_data("sc_raw")

    n_train = SPLIT_SIZES["training"]
    indices = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed))[:args.steps * args.batch_size]
    indices = indices.numpy()
    assert len(np.unique(indices)) == len(indices) == args.steps * args.batch_size
    sets = [RawDataset(data_dir, "training", indices), RawDataset(data_dir, "validation"),
            RawDataset(data_dir, "testing")]
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
               for ds in sets]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = MODELS.build(args.model).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sc_raw] model={args.model} params={n_params:,} train_examples={len(indices)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = task_output("sc_raw", args.model, args.seed)

    history = []
    t0 = time.time()
    model.train()
    loss_sum = seen = 0
    step = 0
    for step, (x, y) in enumerate(loaders[0], 1):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        loss_sum += loss.item() * len(y)
        seen += len(y)
        if step == 1 or step % 128 == 0:
            history.append({"step": step, "train_loss": loss_sum / seen})
            print(f"[sc_raw] step={step} train_loss={loss_sum / seen:.4f}", flush=True)
    val = evaluate(model, loaders[1])
    history.append({"step": step, "train_loss": loss_sum / seen, "val_accuracy": val["accuracy"],
                    "val_loss": val["loss"]})
    torch.save(model.state_dict(), out_dir / "final_model.pt")

    result = {"args": vars(args), "n_params": n_params, "history": history,
              "train_examples_seen": seen, "fraction_of_one_epoch": seen / n_train,
              "val_accuracy": val["accuracy"], "val_loss": val["loss"],
              "total_time_sec": time.time() - t0, "test": None}
    if args.final_test:
        result["test"] = evaluate(model, loaders[2])
    with open(out_dir / "history.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sc_raw] done. steps={step} val_acc={val['accuracy']:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
