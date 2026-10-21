#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# 경로 규칙은 proxy_tasks/common/paths.py 와 같다.
REPO_ROOT="$(cd ../.. && pwd)"
export FFLM_DATA_DIR="${FFLM_DATA_DIR:-${PROXY_DATA_ROOT:-$REPO_ROOT/data}/fflm}"
VENDOR="${PROXY_VENDOR_ROOT:-$REPO_ROOT/third_party}/fflm_vendor"
OUT="${PROXY_OUTPUT_ROOT:-$REPO_ROOT/artifacts}/fflm"
mkdir -p "$OUT"
export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
python - <<'PY'
from pathlib import Path
import hashlib,json,os,torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.__version__.startswith('2.5.1'), 'Expected existing server torch 2.5.1; inspect version before changing protocol'
data=Path(os.environ['FFLM_DATA_DIR'])
m=json.loads((data/'manifest.json').read_text())
for name,s in m['splits'].items():
    assert hashlib.sha256((data/(name+'.npy')).read_bytes()).hexdigest()==s['npy_sha256'], name
print('GPU:',torch.cuda.get_device_name(0),'evaluation data checksums verified',flush=True)
PY
nvidia-smi > "$OUT/gpu_before.txt"
python -m pip freeze > "$OUT/environment.txt"
python run_fflm.py --arch both --smoke --device cuda --batch 2 --eval-batch 2 --out "$OUT/smoke_runs"
python run_fflm.py --arch both --mode benchmark --device cuda --steps 100 --out "$OUT/benchmark_runs"
python run_fflm.py --arch lstm --device cuda --out "$OUT/runs"
python run_fflm.py --arch transformer --device cuda --out "$OUT/runs"
echo "E3 pair completed. See $OUT/runs/*/result.json and $OUT/runs/runs.jsonl."
