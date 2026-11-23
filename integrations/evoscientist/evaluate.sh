#!/usr/bin/env bash
# Shell-tool adapter for the personal harness.
set -euo pipefail
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(realpath -m "${EVO_WORK_ROOT:-$HARNESS/.evo-work}")"
DATA="${EVO_DATA_ROOT:-$HARNESS/data}"
TFL="${EVO_TFL_ROOT:-$HARNESS/third_party/tfl}"
CANDIDATE="$(realpath "$1")"
OUT="$(realpath -m "$2")"
case "$OUT" in "$WORK"/*) ;; *) echo 'Output must remain in the allowed work root' >&2; exit 2;; esac
case "$CANDIDATE" in "$WORK"/*|"$HARNESS"/*) ;; *) echo 'Candidate must remain in an allowed write root' >&2; exit 2;; esac
PROFILE="${EVO_EVAL_PROFILE:-smoke}"
case "$PROFILE" in
  smoke) DEVICE=cpu; TIMEOUT=240 ;;
  full) DEVICE=cuda; TIMEOUT=3600 ;;
  *) echo 'EVO_EVAL_PROFILE must be smoke or full' >&2; exit 2 ;;
esac
BASELINE_ARGS=()
if [[ -n "${EVO_BASELINES:-}" ]]; then
  BASELINE_ARGS=(--baselines "$(realpath "$EVO_BASELINES")")
elif [[ "$PROFILE" == full ]]; then
  echo 'Set EVO_BASELINES to the original full-report baseline JSON before running full mode.' >&2
  exit 2
fi
if [[ "$DEVICE" == cpu ]]; then export CUDA_VISIBLE_DEVICES=''; fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$WORK/tmp" XDG_CACHE_HOME="$WORK/cache"
mkdir -p "$OUT" "$WORK/tmp" "$WORK/cache"
cd "$HARNESS"
if [[ "$PROFILE" == full ]]; then
  export PYTHONPATH="$HARNESS${PYTHONPATH:+:$PYTHONPATH}"
  idle_log="$OUT/gpu_idle.log"
  wait_for_idle_gpu() {
    local started=$SECONDS reading memory utilization busy line
    while true; do
      reading="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1)" || reading="ERROR: $reading"
      printf '%s %s\n' "$(date -Is)" "$reading" >> "$idle_log"
      busy=0
      if [[ "$reading" == ERROR:* || -z "$reading" ]]; then
        busy=1
      else
        while IFS= read -r line; do
          IFS=, read -r memory utilization <<< "$line"
          memory="${memory//[[:space:]]/}"
          utilization="${utilization//[[:space:]]/}"
          if [[ ! "$memory" =~ ^[0-9]+$ || ! "$utilization" =~ ^[0-9]+$ ]] || (( memory > 1024 || utilization > 10 )); then
            busy=1
          fi
        done <<< "$reading"
      fi
      if (( busy == 0 )); then return 0; fi
      if (( SECONDS - started >= 600 )); then
        echo 'GPU did not become idle within 10 minutes' >&2
        return 1
      fi
      sleep 15
    done
  }
  wait_for_idle_gpu
  "${EVO_ML_PYTHON:-python}" -m validation precheck --mixer "$CANDIDATE:CandidateMixer" --data-root "$DATA" --gpu-benchmark --out "$OUT/precheck_gpu.json"
else
  "${EVO_ML_PYTHON:-python}" -m validation precheck --mixer "$CANDIDATE:CandidateMixer" --data-root "$DATA" --out "$OUT/precheck.json"
fi
SUITE=("${EVO_ML_PYTHON:-python}" -m validation suite --mixer "$CANDIDATE:CandidateMixer" --profile "$PROFILE" --seeds 0 --out "$OUT" --data-root "$DATA" --tfl-root "$TFL" --device "$DEVICE" --jobs 1 --timeout "$TIMEOUT")
if [[ "$PROFILE" == smoke ]]; then
  "${SUITE[@]}"
  "${EVO_ML_PYTHON:-python}" -m validation report --suite "$OUT" "${BASELINE_ARGS[@]}"
  exit
fi

# This optional subset is used for integration checks; production runs use all 16 tasks.
if [[ -n "${EVO_EVAL_TASKS:-}" ]]; then
  read -r -a selected_tasks <<< "$EVO_EVAL_TASKS"
  SUITE+=(--tasks "${selected_tasks[@]}")
fi
previous_done=-1
stalled=0
while true; do
  wait_for_idle_gpu
  suite_rc=0
  "${SUITE[@]}" || suite_rc=$?
  read -r done_count total_count < <("${EVO_ML_PYTHON:-python}" - "$OUT/suite.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print('0 0')
else:
    suite = json.loads(p.read_text())
    print(sum(r.get('status') == 'done' for r in suite.get('runs', [])),
          len(suite.get('tasks', [])) * len(suite.get('seeds', [])))
PY
)
  printf '%s done=%s total=%s suite_exit=%s stalled=%s\n' "$(date -Is)" "$done_count" "$total_count" "$suite_rc" "$stalled" >> "$OUT/full_retry.log"
  if (( total_count > 0 && done_count == total_count )); then break; fi
  if (( done_count > previous_done )); then stalled=0; else ((stalled+=1)); fi
  previous_done=$done_count
  if (( stalled >= 3 )); then break; fi
done
if [[ -f "$OUT/suite.json" ]]; then
  "${EVO_ML_PYTHON:-python}" -m validation report --suite "$OUT" "${BASELINE_ARGS[@]}"
fi
if (( total_count == 0 || done_count != total_count )); then
  echo "Full suite incomplete after retries: $done_count/$total_count" >&2
  exit 1
fi
