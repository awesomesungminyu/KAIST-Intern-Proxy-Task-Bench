#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/mad-official-repro
PYTHON=/data/envs/mad-official-repro/bin/python
RUNNER="$ROOT/scripts/run_model.py"
RESULTS="$ROOT/results"
STATUS="$ROOT/status"

mkdir -p \
    "$RESULTS/mh_attention_swiglu" \
    "$RESULTS/striped_mh_hyena_moe" \
    "$STATUS"

rm -f "$STATUS/COMPLETE" "$STATUS/FAILED"

failure_handler() {
    code=$?
    {
        echo "status=failed"
        echo "exit_code=$code"
        date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
    } > "$STATUS/FAILED"
    exit "$code"
}
trap failure_handler ERR INT TERM

date -u +"started_at=%Y-%m-%dT%H:%M:%SZ" \
    > "$STATUS/RUNNING"

echo "=================================================="
echo "MODEL 1/2 — MH Attention + SwiGLU"
echo "Official MAD runs: 396"
echo "=================================================="

PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" \
    --model-id mh_attention_swiglu \
    --layers \
        mh-attention swiglu \
        mh-attention swiglu \
    --output-dir "$RESULTS/mh_attention_swiglu" \
    2>&1 | tee \
        "$RESULTS/mh_attention_swiglu/benchmark.log"

echo "=================================================="
echo "MODEL 2/2 — Striped MH Hyena + MoE"
echo "Official MAD runs: 396"
echo "=================================================="

PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" \
    --model-id striped_mh_hyena_moe \
    --layers \
        mh-hyena moe-mlp \
        mh-attention moe-mlp \
    --output-dir "$RESULTS/striped_mh_hyena_moe" \
    2>&1 | tee \
        "$RESULTS/striped_mh_hyena_moe/benchmark.log"

{
    echo "status=complete"
    date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
} > "$STATUS/COMPLETE"

rm -f "$STATUS/RUNNING" "$STATUS/FAILED"

echo "PASS: BOTH MAD MODELS COMPLETED"
