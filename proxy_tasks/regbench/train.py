"""RegBench 학습/평가 스크립트.

사용:
    python -m regbench.train --seed 0 --model transformer
    python -m regbench.train --smoke_test   # 수십 초짜리 동작 확인용
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.seed import set_seed
from .data import RegBenchDataConfig, RegBenchDataset, collate_fn
from .models import build_model
from .pfa import PFAConfig, vocab_size_with_specials
from common.gpu_guard import require_cuda


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--preset", choices=["paper", "repo"], default="repo")
    p.add_argument("--examples_per_seq", type=int, default=20)
    p.add_argument("--max_total_len", type=int, default=256)
    p.add_argument("--num_train_pfas", type=int, default=None, help="None=매 batch마다 새 DFA")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="runs/regbench")
    p.add_argument("--smoke_test", action="store_true", help="수십 스텝만 돌려 동작 확인")
    p.add_argument("--amp", action="store_true", help="fp16 AMP 사용 (sm_75는 bf16 미지원)")
    return p.parse_args()


def evaluate(model, loader, device, max_batches, use_amp):
    model.eval()
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
                )
            mask = labels != -100
            n_tok = mask.sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
            preds = logits.argmax(-1)
            total_correct += ((preds == labels) & mask).sum().item()
    model.train()
    avg_nll = total_loss / max(total_tokens, 1)
    acc = total_correct / max(total_tokens, 1)
    return {"nll": avg_nll, "bits_per_token": avg_nll / math.log(2), "next_token_acc": acc}


def main():
    args = parse_args()
    if args.smoke_test:
        args.steps = 20
        args.eval_every = 10
        args.eval_steps = 3
        args.batch_size = 8
        args.max_total_len = 64

    set_seed(args.seed)
    device = require_cuda()
    use_amp = args.amp and device == "cuda"

    pfa_cfg = PFAConfig.paper_akyurek2024() if args.preset == "paper" else PFAConfig.repo_default()
    data_cfg = RegBenchDataConfig(
        pfa_cfg=pfa_cfg,
        examples_per_seq=args.examples_per_seq,
        max_total_len=args.max_total_len,
        num_train_pfas=args.num_train_pfas,
        virtual_epoch_size=args.batch_size * args.steps,  # steps 기준으로 충분히 크게
    )

    train_ds = RegBenchDataset(data_cfg, seed=args.seed, split="train")
    test_ds = RegBenchDataset(data_cfg, seed=args.seed, split="test")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    vocab_size = vocab_size_with_specials(pfa_cfg)
    model = build_model(args.model, vocab_size, max_len=args.max_total_len).to(device)
    print(f"[regbench] model={args.model} params={model.num_params():,} device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    model.train()
    t0 = time.time()
    train_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, test_loader, device, args.eval_steps, use_amp)
            elapsed = time.time() - t0
            rec = {"step": step, "train_loss": loss.item(), "elapsed_sec": elapsed, **metrics}
            history.append(rec)
            print(f"[regbench] step={step} train_loss={loss.item():.4f} "
                  f"test_nll={metrics['nll']:.4f} test_acc={metrics['next_token_acc']:.4f} "
                  f"elapsed={elapsed:.1f}s")

    total_time = time.time() - t0
    with open(out_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history, "total_time_sec": total_time}, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[regbench] done. total_time={total_time:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
