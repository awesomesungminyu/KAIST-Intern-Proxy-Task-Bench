#!/usr/bin/env bash
# TITAN RTX 24GB 1장에서 seed 3개를 동시에 띄워보고, 순차 실행 대비 이득이
# 있는지 확인하는 예시. TASK 변수만 바꿔가며 세 task 모두 테스트해볼 것.
#
# 사용법:
#   bash scripts/run_parallel_example.sh regbench transformer
#   bash scripts/run_parallel_example.sh g2p transformer
#   bash scripts/run_parallel_example.sh zinc gine
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-regbench}
MODEL=${2:-transformer}

echo "=== 순차 실행 (seed 0,1,2) ==="
time (
  for s in 0 1 2; do
    python -m "${TASK}.train" --seed "$s" --model "$MODEL" --smoke_test
  done
)

echo ""
echo "=== 동시 실행 (seed 0,1,2 병렬) ==="
time (
  for s in 0 1 2; do
    python -m "${TASK}.train" --seed "$s" --model "$MODEL" --smoke_test &
  done
  wait
)

echo ""
echo "위 두 time 결과를 비교: 병렬 total이 순차 total보다 뚜렷이 작으면"
echo "본 실험도 seed들을 '&'로 동시에 띄우는 게 이득. 비슷하거나 크면 순차 실행."
echo "돌아가는 동안 다른 터미널에서 'watch -n1 nvidia-smi'로 GPU-Util도 같이 볼 것."
