#!/usr/bin/env bash
# 환경 구성 + 데이터 준비.
#   ./setup.sh              # 전체
#   ./setup.sh zinc         # 해당 태스크만
#   ./setup.sh zinc parity  # 여러 개
set -euo pipefail
cd "$(dirname "$0")"

TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(zinc g2p regbench parity tomita5 tomita6 fflm mqar mad_tsc ett scifar sc_raw)
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
  TFL_DIR="${PROXY_VENDOR_ROOT:-$PWD/third_party}/tfl"   # proxy_tasks/common/paths.py 와 같은 규칙
  TFL_PATCH="$PWD/third_party/tfl.patch"
  [ -d "$TFL_DIR" ] || git clone https://github.com/satwik77/Transformer-Formal-Languages.git "$TFL_DIR"
  git -C "$TFL_DIR" checkout -q "$TFL_COMMIT"
  if git -C "$TFL_DIR" apply --check "$TFL_PATCH" 2>/dev/null; then
    git -C "$TFL_DIR" apply "$TFL_PATCH" && echo "[setup]   패치 적용"
  else
    echo "[setup]   패치 이미 적용됨 (또는 third_party/tfl.patch 확인 필요)"
  fi
fi

VENDOR_DIR="${PROXY_VENDOR_ROOT:-$PWD/third_party}"
if want mqar; then
  echo "[setup] mqar — zoology 고정 커밋 클론 (데이터는 실행 시 생성)"
  ZOOLOGY_COMMIT=$(awk '{print $3}' third_party/zoology.commit)
  [ -d "$VENDOR_DIR/zoology" ] || git clone https://github.com/HazyResearch/zoology.git "$VENDOR_DIR/zoology"
  git -C "$VENDOR_DIR/zoology" checkout -q "$ZOOLOGY_COMMIT"
  echo "[setup]   mqar 는 pydantic·wandb·einops 가 있는 torch 2.2.2 환경이 필요하다 (protocol runtime)"
fi

if want mad_tsc; then
  echo "[setup] mad_tsc — mad-lab 고정 커밋 클론 + Turing SDPA 패치"
  MAD_COMMIT=$(awk '{print $3}' third_party/mad.commit)
  MAD_PATCH="$PWD/third_party/mad-turing-sdpa.patch"
  [ -d "$VENDOR_DIR/mad-lab" ] || git clone https://github.com/athms/mad-lab.git "$VENDOR_DIR/mad-lab"
  git -C "$VENDOR_DIR/mad-lab" checkout -q "$MAD_COMMIT"
  if git -C "$VENDOR_DIR/mad-lab" apply --check "$MAD_PATCH" 2>/dev/null; then
    git -C "$VENDOR_DIR/mad-lab" apply "$MAD_PATCH" && echo "[setup]   패치 적용"
  else
    echo "[setup]   패치 이미 적용됨 (또는 third_party/mad-turing-sdpa.patch 확인 필요)"
  fi
  echo "[setup]   mad_tsc 는 pytorch-lightning 2.2.1 이 있는 torch 2.2.2 환경이 필요하다 (protocol runtime)"
fi

want fflm && echo "[setup] fflm — data/vendor는 Git에 포함되지 않습니다. scripts/restore_fflm_assets.py 가 \$PROXY_DATA_ROOT/fflm 과 \$PROXY_VENDOR_ROOT/fflm_vendor 로 복원합니다 (docs/EVOSCIENTIST.md)."

echo "[setup] 완료 — 예: ./run.sh zinc gine 0"
