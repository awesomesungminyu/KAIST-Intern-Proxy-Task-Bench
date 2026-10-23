#!/usr/bin/env bash
# 통합 실행 진입점.
#   ./run.sh zinc|g2p|regbench <model> <seed> [추가 인자...]
#   ./run.sh parity|tomita5|tomita6 [pilot|full|summary]
#   ./run.sh fflm
set -euo pipefail
echo "[legacy] run.sh does not record validation provenance. Prefer python -m validation (README.md)." >&2
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # gpu_guard 가 빈 값을 거부한다

TASK=${1:-}; shift || true
case "$TASK" in
  zinc|g2p|regbench)
    MODEL=${1:?"사용법: ./run.sh $TASK <model> <seed> [...]"}
    SEED=${2:?"사용법: ./run.sh $TASK <model> <seed> [...]"}; shift 2
    # 산출물 위치는 train.py 기본값(common/paths.py: $PROXY_OUTPUT_ROOT/<task>/<model>-s<seed>)을 따른다.
    echo "[run] $TASK model=$MODEL seed=$SEED"
    cd proxy_tasks
    exec python -m "${TASK}.train" --model "$MODEL" --seed "$SEED" "$@"
    ;;
  parity|tomita5|tomita6)
    TFL_DIR="${PROXY_VENDOR_ROOT:-$PWD/third_party}/tfl"
    [ -d "$TFL_DIR" ] || { echo "$TFL_DIR 없음 — 먼저 ./setup.sh $TASK"; exit 1; }
    case "$TASK" in
      parity) SCRIPT=proxy_tasks/formal_languages/run_parity_mini.sh ;;
      *)      SCRIPT=proxy_tasks/formal_languages/run_${TASK}.sh ;;
    esac
    exec env REPO="$TFL_DIR" bash "$SCRIPT" "${1:-full}"
    ;;
  fflm) exec bash proxy_tasks/fflm/start_server.sh ;;
  *)
    echo "사용법: ./run.sh <task> ..."
    echo "  task: zinc | g2p | regbench | parity | tomita5 | tomita6 | fflm"
    exit 1 ;;
esac
