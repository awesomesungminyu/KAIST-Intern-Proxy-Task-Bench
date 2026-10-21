#!/usr/bin/env bash
# 세 task 모두 수십 초~수 분 안에 끝나는 스모크 테스트.
# GPU/환경/코드 경로가 다 맞는지 먼저 이걸로 확인한 다음 본 실험 돌릴 것.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT=$(python -c "from common.paths import DATA_ROOT; print(DATA_ROOT)")

echo "=== [1/3] RegBench smoke test ==="
python -m regbench.train --smoke_test

echo "=== [2/3] G2P smoke test ==="
if [ ! -f "$DATA_ROOT/g2p/vocab.json" ]; then
  python -m g2p.prepare_data
fi
python -m g2p.train --smoke_test

echo "=== [3/3] ZINC smoke test ==="
if [ ! -f "$DATA_ROOT/zinc/metadata.json" ]; then
  python -m zinc.prepare_data
fi
python -m zinc.train --smoke_test

echo "=== 모든 스모크 테스트 통과 ==="
