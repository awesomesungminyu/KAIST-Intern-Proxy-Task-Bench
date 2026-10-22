"""문자열 전사 + 길이 일반화 학습/평가 스크립트 (Jelassi et al. 2024 축소판).

사용 (proxy_tasks/ 에서):
    python -m jelassi_copy.train --model transformer --seed 0
    python -m jelassi_copy.train --model mamba --smoke_test
학습은 길이 [min_len, max_len] 문자열만 본다. validation 은 EVAL_SEED 로 고정한 문자열을
학습 구간(id)과 학습 구간 밖(ood_*) 으로 나눠 잰다. 별도 test split 은 없다 (생성 분포가 같다).
최적화는 공식 train_utils.py 와 같은 형태: AdamW(weight_decay 0.1), warmup 100 + linear decay, clip 1.0.
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

from common.gpu_guard import require_cuda
from common.paths import task_output
from common.seed import set_seed
from .data import VOCAB, eval_strings, pad_batch, score, train_batch
from .models import MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=MODELS.names(), default="transformer")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--context_len", type=int, default=220, help="공식 코드 기본값")
    p.add_argument("--min_len", type=int, default=5)
    p.add_argument("--max_len", type=int, default=50, help="L_train. 이보다 긴 평가 구간이 OOD")
    p.add_argument("--ood_bins", type=str, default="51-100,101-200,201-300")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_per_bin", type=int, default=256)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--out_dir", type=str, default=None, help="기본: $PROXY_OUTPUT_ROOT/jelassi_copy/<model>-s<seed>")
    p.add_argument("--smoke_test", action="store_true", help="20 step, 구간당 8 문자열")
    return p.parse_args()


def bins_for(args):
    bins = [("id", args.min_len, args.max_len)]
    for spec in args.ood_bins.split(","):
        lo, hi = (int(v) for v in spec.split("-"))
        if lo <= args.max_len:
            raise ValueError(f"OOD bin {spec} overlaps the training range")
        bins.append((f"ood_{lo}_{hi}", lo, hi))
    return bins


@torch.no_grad()
def evaluate(model, strings, batch):
    model.eval()
    out = {}
    for name, examples in strings.items():
        exact, chars = [], []
        for i in range(0, len(examples), batch):
            group = examples[i:i + batch]
            tokens = pad_batch(group).cuda()
            logits = model(tokens)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation output")
            e, c = score(logits, tokens, [(len(g) - 3) // 2 for g in group])
            exact += e
            chars += c
        out[name] = {"exact_match": float(np.mean(exact)), "char_accuracy": float(np.mean(chars)), "n": len(exact)}
    model.train()
    ood = [v["exact_match"] for k, v in out.items() if k.startswith("ood_")]
    return {"bins": out, "id_exact_match": out["id"]["exact_match"], "ood_exact_match": float(np.mean(ood))}


def main():
    args = parse_args()
    if args.smoke_test:
        args.steps, args.eval_every, args.eval_per_bin = 20, 10, 8
    bins = bins_for(args)
    device = require_cuda("[jelassi_copy]")
    set_seed(args.seed)

    strings = eval_strings(bins, args.eval_per_bin)
    model = MODELS.build(args.model, vocab_size=VOCAB).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[jelassi_copy] model={args.model} params={n_params:,} device={torch.cuda.get_device_name(0)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / args.warmup,
                                                                 max(0.0, (args.steps - s) / max(1, args.steps - args.warmup))))
    rng = np.random.default_rng(args.seed)

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = task_output("jelassi_copy", args.model, args.seed)

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        tokens, mask = (t.to(device, non_blocking=True) for t in
                        train_batch(rng, args.batch_size, args.context_len, args.min_len, args.max_len))
        logits = model(tokens[:, :-1])
        loss = (F.cross_entropy(logits.reshape(-1, VOCAB), tokens[:, 1:].reshape(-1), reduction="none")
                * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, strings, args.eval_batch)
            rec = {"step": step, "train_loss": loss.item(), "elapsed_sec": time.time() - t0, **metrics}
            history.append(rec)
            per_bin = " ".join(f"{k}={v['exact_match']:.3f}" for k, v in metrics["bins"].items())
            print(f"[jelassi_copy] step={step} train_loss={loss.item():.4f} {per_bin} "
                  f"elapsed={rec['elapsed_sec']:.1f}s", flush=True)

    final = history[-1]
    result = {"args": vars(args), "n_params": n_params, "device": torch.cuda.get_device_name(0),
              "bins": [{"name": n, "min_len": lo, "max_len": hi} for n, lo, hi in bins],
              "history": history, "id_exact_match": final["id_exact_match"],
              "ood_exact_match": final["ood_exact_match"], "final_bins": final["bins"],
              "total_time_sec": time.time() - t0}
    with open(out_dir / "history.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[jelassi_copy] done. id={final['id_exact_match']:.3f} ood={final['ood_exact_match']:.3f} -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
