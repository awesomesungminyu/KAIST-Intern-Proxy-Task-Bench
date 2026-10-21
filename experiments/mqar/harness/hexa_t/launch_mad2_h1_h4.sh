#!/usr/bin/env bash
cd /data/mqar

while pgrep -f '[c]ompare_mad2_h1\.py' >/dev/null 2>&1; do
    echo "$(date -Is) waiting for the existing H1 comparison"
    sleep 30
done

export PYTHONPATH=/data/mqar/vendor/mad-lab
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

/data/envs/hexa-mad/bin/python -u \
    harness/hexa_t/compare_mad2_h1_h4.py \
    >> results/hexa_t/mad2_h1_comparison/h1_h4_queue.log 2>&1
