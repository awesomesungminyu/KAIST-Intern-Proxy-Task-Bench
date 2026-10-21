"""G2P 학습/평가 스크립트.

사용:
    python -m g2p.prepare_data          # 최초 1회
    python -m g2p.train --seed 0 --model transformer
    python -m g2p.train --smoke_test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.seed import set_seed
from .data import G2PDataset, collate_fn, load_split, load_vocab, BOS_ID, EOS_ID, PAD_ID
from .models import build_model
from common.gpu_guard import require_cuda

DATA_DIR = Path(__file__).parent / "data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_len", type=int, default=40)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="runs/g2p")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def greedy_decode(model, src, max_len, device, model_name):
    """Teacher-forcing 없이 한 글자씩 생성해서 단어 정확도(WER) 계산용."""
    was_training = model.training
    model.eval()
    B = src.size(0)
    tgt = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = model(src, tgt)
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
            next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, PAD_ID), next_tok)
            tgt = torch.cat([tgt, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == EOS_ID)
            if finished.all():
                break
    model.train(was_training)
    return tgt


def sequence_exact_match(pred: torch.Tensor, target_out: torch.Tensor) -> float:
    """전체 시퀀스(EOS까지) 완전 일치 비율 = word accuracy."""
    B = pred.size(0)
    correct = 0
    for i in range(B):
        p = pred[i, 1:].tolist()  # BOS 제외
        t = target_out[i].tolist()
        # -100(ignore) 이전까지, EOS 포함해서 비교
        t_trim = []
        for x in t:
            if x == -100:
                break
            t_trim.append(x)
        p_trim = []
        for x in p:
            p_trim.append(x)
            if x == EOS_ID:
                break
        if p_trim == t_trim:
            correct += 1
    return correct / B


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.batch_size = 16

    set_seed(args.seed)
    device = require_cuda()
    use_amp = args.amp and device == "cuda"

    data_dir = Path(args.data_dir)
    char_vocab, phone_vocab = load_vocab(data_dir / "vocab.json")
    train_entries = load_split(data_dir / "train.tsv")
    val_entries = load_split(data_dir / "val.tsv")
    if args.smoke_test:
        train_entries = train_entries[:500]
        val_entries = val_entries[:100]

    train_ds = G2PDataset(train_entries, char_vocab, phone_vocab)
    val_ds = G2PDataset(val_entries, char_vocab, phone_vocab)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = build_model(args.model, len(char_vocab), len(phone_vocab), args.max_len).to(device)
    print(f"[g2p] model={args.model} params={model.num_params():,} device={device} "
          f"train={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(src, tgt_in)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=-100
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        # 검증: teacher-forcing loss + greedy-decode word accuracy
        model.eval()
        val_loss, val_batches, val_acc_sum, val_n = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                tgt_in = batch["tgt_in"].to(device)
                tgt_out = batch["tgt_out"].to(device)
                logits = model(src, tgt_in)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=-100
                )
                val_loss += loss.item()
                val_batches += 1

                pred = greedy_decode(model, src, args.max_len, device, args.model)
                acc = sequence_exact_match(pred, tgt_out)
                val_acc_sum += acc * src.size(0)
                val_n += src.size(0)
        model.train()

        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "val_loss": val_loss / max(val_batches, 1),
            "val_word_acc": val_acc_sum / max(val_n, 1),
            "elapsed_sec": elapsed,
        }
        history.append(rec)
        print(f"[g2p] epoch={epoch} train_loss={rec['train_loss']:.4f} "
              f"val_loss={rec['val_loss']:.4f} val_word_acc={rec['val_word_acc']:.4f} "
              f"elapsed={elapsed:.1f}s")

    total_time = time.time() - t0
    with open(out_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history, "total_time_sec": total_time}, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[g2p] done. total_time={total_time:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
