#!/usr/bin/env bash
# 환경 구성 + 데이터 준비.
#   ./setup.sh              # 전체
#   ./setup.sh zinc         # 해당 태스크만
#   ./setup.sh zinc parity  # 여러 개
set -euo pipefail
cd "$(dirname "$0")"

TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(zinc g2p regbench parity tomita5 tomita6 fflm)
want() { printf '%s\n' "${TASKS[@]}" | grep -qx "$1"; }

PY=${PYTHON:-python3}
[ -d .venv ] || { echo "[setup] venv 생성"; "$PY" -m venv .venv; }
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] 공통 의존성"
pip install --upgrade pip -q
# 플랫폼별 torch: 실험 서버는 CUDA 12.4, macOS 는 CPU/MPS 빌드(편집·분석용).
if [ "$(uname -s)" = "Darwin" ]; then
  echo "[setup] macOS 감지 — CPU/MPS 빌드 설치 (학습은 CUDA 서버에서)"
  pip install torch==2.5.1
else
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
fi
pip install -r requirements.txt
python -c "import torch; print('[setup] torch', torch.__version__, '| cuda', torch.cuda.is_available())"

if want zinc; then
  echo "[setup] zinc — PyG + 데이터 (수 분, 수백 MB 다운로드)"
  pip install torch-geometric==2.8.0
  (cd proxy_tasks && python -m zinc.prepare_data)
fi

if want g2p; then
  echo "[setup] g2p — 데이터"
  (cd proxy_tasks && python -m g2p.prepare_data)
fi

want regbench && echo "[setup] regbench — 데이터는 실행 시 생성된다 (준비 불필요)"

if want parity || want tomita5 || want tomita6; then
  echo "[setup] parity/tomita — 외부 저장소 클론 + 패치"
  TFL_COMMIT=$(awk '{print $3}' third_party/tfl.commit)
  [ -d third_party/tfl ] || git clone https://github.com/satwik77/Transformer-Formal-Languages.git third_party/tfl
  git -C third_party/tfl checkout -q "$TFL_COMMIT"
  if git -C third_party/tfl apply --check ../tfl.patch 2>/dev/null; then
    git -C third_party/tfl apply ../tfl.patch && echo "[setup]   패치 적용"
  else
    echo "[setup]   패치 이미 적용됨 (또는 third_party/tfl.patch 확인 필요)"
  fi
fi

want fflm && echo "[setup] fflm — 의존성은 vendor 되어 있고 데이터(약 51MB)는 실행 시 생성된다"

echo "[setup] 완료 — 예: ./run.sh zinc gine 0"
