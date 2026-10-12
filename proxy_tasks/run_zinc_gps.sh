#!/usr/bin/env bash
# ZINC-12K GraphGPS 본 실험: 3 seed x 500 epoch, 순차 단독 실행(시간 측정 오염 방지).
# ckpt_epoch=200 -> 같은 학습 경로에서 200/500 결과를 모두 추출.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
H=60; HEADS=4
for s in 0 1 2; do
  echo "=== seed $s START $(date -Is) ==="
  python -m zinc.train --model gps --seed "$s" --epochs 500 \
    --hidden_dim $H --heads $HEADS --ckpt_epoch 200 \
    --out_dir "runs/zinc_gps_s${s}_e500" \
    > "logs/zinc_gps/gps_s${s}_e500.log" 2>&1
  echo "=== seed $s EXIT=$? $(date -Is) ==="
done
echo "ALL DONE $(date -Is)"
