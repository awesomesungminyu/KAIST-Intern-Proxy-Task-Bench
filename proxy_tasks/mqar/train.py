"""MQAR(multi-query associative recall) 학습/평가 스크립트.

사용 (proxy_tasks/ 에서, zoology 가 import 되는 환경에서):
    python -m mqar.train --model transformer --seed 456
    python -m mqar.train --model gqa2 --smoke_test
데이터는 zoology 의 MQAR 생성기가 실행 시 만든다 (third_party/zoology 고정 커밋).
학습 설정은 experiments/mqar/train_mqar_baseline.py 의 CONFIG 와 같다: 최대 30 epoch,
validation 정확도 0.99 도달 시 종료. 별도 test split 은 없다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common.gpu_guard import require_cuda
from common.paths import THIRD_PARTY, task_output
from common.seed import set_seed
from .models import MODELS, TransformerLM

CONFIG = {
    "vocab_size": 8192,
    "sequence_length": 64,
    "num_kv_pairs": 8,
    "train_examples": 20_000,
    "valid_examples": 2_000,
    "batch_size": 64,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "ffn_dim": 512,
    "learning_rate": 1e-3,
    "weight_decay": 0.1,
    "max_epochs": 30,
    "early_stop_accuracy": 0.99,
    "data_seed_train": 123,
    "data_seed_valid": 2_147_483_789,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=456, help="모델 초기화·데이터 순서 seed (원 실행: 456, 789, 101112)")
    p.add_argument("--model", choices=MODELS.names(), default="transformer")
    p.add_argument("--max_epochs", type=int, default=CONFIG["max_epochs"])
    p.add_argument("--zoology_root", type=str, default=None, help="기본: $PROXY_VENDOR_ROOT/zoology")
    p.add_argument("--out_dir", type=str, default=None, help="기본: $PROXY_OUTPUT_ROOT/mqar/<model>-s<seed>")
    p.add_argument("--smoke_test", action="store_true", help="1 epoch, train 256 / valid 128 예제")
    return p.parse_args()


def make_dataset(config, num_examples, seed):
    from zoology.data.multiquery_ar import MQARConfig

    data = MQARConfig(
        vocab_size=config["vocab_size"],
        num_examples=num_examples,
        input_seq_len=config["sequence_length"],
        num_kv_pairs=config["num_kv_pairs"],
        random_non_queries=True,
        include_slices=True,
    ).build(seed=seed)
    return TensorDataset(data.inputs, data.labels)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum = 0.0
    correct = target_count = 0
    for inputs, labels in loader:
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        logits = model(inputs)
        mask = labels != -100
        targets, selected = labels[mask], logits[mask]
        loss_sum += F.cross_entropy(selected, targets, reduction="sum").item()
        correct += (selected.argmax(dim=-1) == targets).sum().item()
        target_count += targets.numel()
    return loss_sum / target_count, correct / target_count


def main():
    args = parse_args()
    config = dict(CONFIG, model_seed=args.seed, max_epochs=args.max_epochs)
    if args.smoke_test:
        config.update(train_examples=256, valid_examples=128, max_epochs=1)
    sys.path.insert(0, str(Path(args.zoology_root) if args.zoology_root else THIRD_PARTY / "zoology"))
    require_cuda("[mqar]")
    set_seed(args.seed)

    train_dataset = make_dataset(config, config["train_examples"], config["data_seed_train"])
    valid_dataset = make_dataset(config, config["valid_examples"], config["data_seed_valid"])
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              generator=loader_generator, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=True)

    model = TransformerLM(args.model, config).to(device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mqar] model={args.model} params={n_params:,} torch={torch.__version__}", flush=True)

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = task_output("mqar", args.model, args.seed)

    t0 = time.time()
    best_accuracy, best_loss, best_epoch = -1.0, None, 0
    global_step = 0
    history = []
    stop = "max_epochs"
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = train_targets = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            mask = labels != -100
            targets, selected = labels[mask], logits[mask]
            loss = F.cross_entropy(selected, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = targets.numel()
            train_loss_sum += loss.item() * count
            train_correct += (selected.argmax(dim=-1) == targets).sum().item()
            train_targets += count
            global_step += 1
        valid_loss, valid_accuracy = evaluate(model, valid_loader)
        history.append({"epoch": epoch, "global_step": global_step, "train_loss": train_loss_sum / train_targets,
                        "train_accuracy": train_correct / train_targets, "valid_loss": valid_loss,
                        "valid_accuracy": valid_accuracy})
        print(f"[mqar] epoch={epoch:02d} valid_loss={valid_loss:.5f} valid_acc={valid_accuracy:.4f}", flush=True)
        if valid_accuracy > best_accuracy:
            best_accuracy, best_loss, best_epoch = valid_accuracy, valid_loss, epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        if valid_accuracy >= config["early_stop_accuracy"]:
            stop = "early_stopping"
            break

    result = {"args": vars(args), "config": config, "n_params": n_params, "history": history,
              "best_epoch": best_epoch, "best_valid_accuracy": best_accuracy, "best_valid_loss": best_loss,
              "passed": best_accuracy >= config["early_stop_accuracy"], "stop_reason": stop,
              "torch": torch.__version__, "total_time_sec": time.time() - t0}
    with open(out_dir / "history.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[mqar] done. best_valid_accuracy={best_accuracy:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
