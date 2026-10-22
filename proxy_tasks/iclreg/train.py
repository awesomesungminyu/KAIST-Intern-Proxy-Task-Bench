"""In-context linear regression 학습/평가 스크립트 (Garg et al. 2022 축소판).

사용 (proxy_tasks/ 에서):
    python -m iclreg.train --model transformer --seed 0
    python -m iclreg.train --model lstm --smoke_test
학습 데이터는 매 step 새로 생성한다 (seed 고정). validation 은 EVAL_SEED 로 고정한 별도 프롬프트이며
모든 모델·seed 가 같은 프롬프트를 본다. 별도 test split 은 없다 (생성 분포가 같다).
Garg et al. 의 curriculum (차원 5→20, 점 11→41, 2000 step 마다 +1/+2) 을 그대로 쓴다.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from common.gpu_guard import require_cuda
from common.paths import task_output
from common.seed import set_seed
from .data import baselines, eval_set, normalized_error_by_k, sample
from .models import MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=MODELS.names(), default="transformer")
    p.add_argument("--n_dims", type=int, default=20)
    p.add_argument("--n_points", type=int, default=41, help="프롬프트당 (x,y) 수. query 직전 예제 수 k=0..n_points-1")
    p.add_argument("--steps", type=int, default=45000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--curriculum_start_dims", type=int, default=5)
    p.add_argument("--curriculum_start_points", type=int, default=11)
    p.add_argument("--curriculum_interval", type=int, default=2000, help="이 step 마다 차원 +1, 점 +2")
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--eval_prompts", type=int, default=2048)
    p.add_argument("--out_dir", type=str, default=None, help="기본: $PROXY_OUTPUT_ROOT/iclreg/<model>-s<seed>")
    p.add_argument("--smoke_test", action="store_true", help="20 step, 작은 validation")
    return p.parse_args()


def curriculum(args, step):
    """Garg et al. 형: (step-1)//interval 번 증가한 (active dims, points)."""
    n = (step - 1) // args.curriculum_interval
    return (min(args.n_dims, args.curriculum_start_dims + n),
            min(args.n_points, args.curriculum_start_points + 2 * n))


@torch.no_grad()
def evaluate(model, xs, ys, n_dims, batch=256):
    model.eval()
    pred = torch.cat([model(xs[i:i + batch].cuda(), ys[i:i + batch].cuda()).float().cpu()
                      for i in range(0, len(xs), batch)])
    model.train()
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Nonfinite evaluation output")
    by_k = normalized_error_by_k(pred, ys, n_dims)
    return {"val_mse_by_k": by_k.tolist(), "val_k_ge_d_mse": by_k[n_dims:].mean().item(),
            "val_k_lt_d_mse": by_k[:n_dims].mean().item()}


def main():
    args = parse_args()
    if args.smoke_test:
        args.steps, args.eval_every, args.eval_prompts = 20, 10, 64
    if args.n_points <= args.n_dims:
        raise ValueError("n_points must exceed n_dims so that k>=d is evaluated")
    device = require_cuda("[iclreg]")
    set_seed(args.seed)

    xs_val, ys_val = eval_set(args.eval_prompts, args.n_points, args.n_dims)
    reference = baselines(xs_val, ys_val, args.n_dims)
    model = MODELS.build(args.model, n_dims=args.n_dims, n_points=args.n_points).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[iclreg] model={args.model} params={n_params:,} device={torch.cuda.get_device_name(0)}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = task_output("iclreg", args.model, args.seed)

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        dims, points = curriculum(args, step)
        xs, ys = sample(args.batch_size, points, args.n_dims, gen, device, active_dims=dims)
        loss = ((model(xs, ys) - ys) ** 2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, xs_val, ys_val, args.n_dims)
            rec = {"step": step, "train_loss": loss.item(), "curriculum_dims": dims, "curriculum_points": points,
                   "elapsed_sec": time.time() - t0, **metrics}
            history.append(rec)
            print(f"[iclreg] step={step} train_loss={loss.item():.4f} dims={dims} points={points} "
                  f"val_k<d={metrics['val_k_lt_d_mse']:.4f} val_k>=d={metrics['val_k_ge_d_mse']:.4f} "
                  f"elapsed={rec['elapsed_sec']:.1f}s", flush=True)

    final = history[-1]
    result = {"args": vars(args), "n_params": n_params, "device": torch.cuda.get_device_name(0),
              "eval_seed_prompts": args.eval_prompts, "history": history,
              "val_k_ge_d_mse": final["val_k_ge_d_mse"], "val_k_lt_d_mse": final["val_k_lt_d_mse"],
              "val_mse_by_k": final["val_mse_by_k"], **reference,
              "least_squares_k_ge_d_mse": sum(reference["least_squares_by_k"][args.n_dims:]) / (args.n_points - args.n_dims),
              "total_time_sec": time.time() - t0}
    with open(out_dir / "history.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[iclreg] done. val_k>=d={final['val_k_ge_d_mse']:.4f} "
          f"least_squares_k>=d={result['least_squares_k_ge_d_mse']:.2e} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
