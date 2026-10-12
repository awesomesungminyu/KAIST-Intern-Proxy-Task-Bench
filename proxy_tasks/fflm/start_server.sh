#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONPATH="$PWD/vendor${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
python - <<'PY'
from pathlib import Path
import hashlib,json,torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.__version__.startswith('2.5.1'), 'Expected existing server torch 2.5.1; inspect version before changing protocol'
m=json.loads(Path('data/manifest.json').read_text())
for name,s in m['splits'].items():
    assert hashlib.sha256(Path('data',name+'.npy').read_bytes()).hexdigest()==s['npy_sha256'], name
print('GPU:',torch.cuda.get_device_name(0),'evaluation data checksums verified',flush=True)
PY
nvidia-smi > gpu_before.txt
python -m pip freeze > environment.txt
python run_fflm.py --arch both --smoke --device cuda --batch 2 --eval-batch 2 --out smoke_runs
python run_fflm.py --arch both --mode benchmark --device cuda --steps 100 --out benchmark_runs
python run_fflm.py --arch lstm --device cuda --out runs
python run_fflm.py --arch transformer --device cuda --out runs
echo 'E3 pair completed. See runs/*/result.json and runs/runs.jsonl.'
