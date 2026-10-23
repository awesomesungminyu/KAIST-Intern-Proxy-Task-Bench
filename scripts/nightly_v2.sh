#!/usr/bin/env bash
# Overnight protocol-v2 full runs (common architecture set) through scripts/validation_v2.py.
#
#   scripts/nightly_v2.sh --out DIR --cells FILE [--worktree DIR] [--deadline TIME] [--seeds "0 1 2"]
#                         [--smoke-status FILE] [--dry-run [--simulate FILE]]
#
#   --cells FILE    one cell per line: "task model timeout_s" (# comments allowed); order is kept within each
#                   group below. Each seed runs: other cells, risk cells (any cs_mamba, scifar cs_gla), then fflm
#                   (its non-risk cells, then its risk cells: cs_mamba, cs_linear_attention).
#   --deadline      anything `date -d` accepts; no new cell starts after it (logs DEADLINE, exit 0).
#   --dry-run       print checks, cell order and commands; nothing runs, no GPU is touched.
#   --simulate FILE with --dry-run: lines "seed task model outcome,outcome,..." (outcomes per attempt:
#                   done | timeout | failed | oom | cuda_init) drive the retry logic instead of real runs.
#
# Order: all cells of seed 0, then seed 1, then seed 2. One cell at a time. After the last cell of a seed:
# SEED_DONE in the progress log and OUT/summary_seed<s>.{csv,md} (plus the running OUT/summary.{csv,md}).
# Before each attempt: wait for 10 consecutive idle seconds (memory <= 1024 MiB, utilisation <= 10 %), max 30 min.
# Retry rules (same as the v2 smoke driver):
#   CUDA init failure (died within 10 s with "CUDA driver initialization failed"): check the GPU (nvidia-smi,
#     torch.zeros(1).cuda()), wait 30-60 s, retry the same cell, at most 3 retries; then status cuda_init_failed
#     and the next cell. 3 consecutive cuda_init_failed cells: ABORT (nvidia-smi and the error in OUT/ABORT).
#   OOM: wait 60 s, 1 retry.  Other failure: 1 retry.  timeout: recorded, no retry.
#   Every attempt uses a new --out directory (the runner refuses an existing one).
# Refuses to start if the worktree is not clean, if OUT already holds a run, or if any listed cell is not
# "done" in the smoke status TSV (the reason is printed). Code and protocol files are only read.
set +e
WT=/workspace/proxy-bench-commonset-v2
OUT= CELLS= DEADLINE= SEEDS="0 1 2" SMOKE= DRY=0 SIM=
while [ $# -gt 0 ]; do
  case $1 in
    --worktree) WT=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --cells) CELLS=$2; shift 2 ;;
    --deadline) DEADLINE=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --smoke-status) SMOKE=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --simulate) SIM=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] && [ -n "$CELLS" ] || { echo "--out and --cells are required" >&2; exit 2; }
[ -n "$SIM" ] && [ $DRY -eq 0 ] && { echo "--simulate needs --dry-run" >&2; exit 2; }
SMOKE=${SMOKE:-$WT/artifacts/v2_smoke_20261023/smoke_status.tsv}
PY=/opt/conda/bin/python
DEADLINE_S=
if [ -n "$DEADLINE" ]; then DEADLINE_S=$(date -d "$DEADLINE" +%s) || { echo "bad --deadline: $DEADLINE" >&2; exit 2; }; fi

export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PROXY_DATA_ROOT=/data/proxy-task-bench/data
export PROXY_VENDOR_ROOT=/workspace/proxy-task-bench/third_party      # read only (zoology, fflm_vendor)
export PROXY_OUTPUT_ROOT=$OUT/output
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$OUT/triton_cache}       # keep compile caches out of $HOME

STATUS=$OUT/status.tsv          # one row per cell (final attempt), with attempts
ATTEMPTS=$OUT/attempts.tsv      # one row per attempt
PROG=$OUT/progress.log
CUDA_RE='CUDA driver initialization failed'
OOM_RE='out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED'
COLS="seed\ttask\tmodel\tattempts\tstatus\tclass\tstart\tend\twall_s\ttimeout_s\tgpu_free_MiB_before\tscore\tmetric\tout\terror"

say() { echo "$*"; }
log() { echo "$* $(date -Is)" >> $PROG; say "$*"; }
field() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]));print($2)" "$1" 2>/dev/null; }
is_risk() { case "$1:$2" in *:cs_mamba|scifar:cs_gla|fflm:cs_linear_attention) return 0 ;; esac; return 1; }

# ---------------------------------------------------------------- start checks
REFUSE=()
if [ $DRY -eq 0 ]; then
  for f in $STATUS $PROG $OUT/cells; do [ -e $f ] && REFUSE+=("$f exists (use a new --out)"); done
fi
[ -z "$(git -C $WT status --porcelain 2>&1)" ] || REFUSE+=("worktree $WT is not clean (git status --porcelain)")
PLAN=$(grep -vE '^\s*(#|$)' $CELLS)
[ -n "$PLAN" ] || REFUSE+=("no cells in $CELLS")
if [ -f "$SMOKE" ]; then
  NOT_DONE=$($PY - "$SMOKE" "$CELLS" <<'EOF'
import csv, sys
last = {}
for r in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
    last[(r["task"], r["model"])] = r["status"]
for line in open(sys.argv[2]):
    if line.strip() and not line.lstrip().startswith("#"):
        task, model = line.split()[:2]
        if last.get((task, model)) != "done":
            print(f"{task}:{model}={last.get((task, model), 'missing')}")
EOF
)
  [ -z "$NOT_DONE" ] || REFUSE+=("smoke not passed for: $(echo $NOT_DONE)")
else
  REFUSE+=("smoke status file $SMOKE missing")
fi
if [ ${#REFUSE[@]} -gt 0 ]; then
  for r in "${REFUSE[@]}"; do echo "refusing to start: $r" >&2; done
  [ $DRY -eq 1 ] && echo "[dry-run] continuing despite the refusals above to show the plan" >&2 || exit 1
fi

# ---------------------------------------------------------------- ordered cell list
group() {  # task model -> 0 other, 1 risk, 2 fflm, 3 fflm risk
  local g=0; is_risk $1 $2 && g=1; [ $1 = fflm ] && g=$((g + 2)); echo $g
}
ORDER=()
for s in $SEEDS; do
  for g in 0 1 2 3; do
    while read -r t m to; do [ $(group $t $m) = $g ] && ORDER+=("$s $t $m $to"); done <<< "$PLAN"
  done
done

mkdir -p $OUT/cells
[ -f $STATUS ] || echo -e "$COLS" > $STATUS
[ -f $ATTEMPTS ] || echo -e "$COLS" > $ATTEMPTS

# ---------------------------------------------------------------- helpers
wait_idle() {   # 10 consecutive idle seconds; give up after 30 min (the runner then refuses a busy GPU)
  if [ $DRY -eq 1 ]; then say "  [dry-run] wait for GPU idle (<=1024 MiB, <=10 %, 10 s, max 30 min)"; return; fi
  local ok=0 waited=0 mem util
  while [ $ok -lt 10 ] && [ $waited -lt 1800 ]; do
    read -r mem util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if [ "$mem" -le 1024 ] && [ "$util" -le 10 ]; then ok=$((ok+1)); else ok=0; fi
    sleep 1; waited=$((waited+1))
  done
  [ $ok -ge 10 ] || log "WAIT_TIMEOUT GPU not idle after 1800s"
}

gpu_check() {   # file
  if [ $DRY -eq 1 ]; then say "  [dry-run] GPU check: nvidia-smi; python -c 'torch.zeros(1).cuda()' -> $1"; return; fi
  { echo "--- $(date -Is) nvidia-smi"; nvidia-smi 2>&1
    echo "--- torch.zeros(1).cuda()"; timeout 120 $PY -c "import torch; print(torch.zeros(1).cuda().item())" 2>&1; echo "rc=$?"; } > $1
}

sim_outcome() {   # seed task model attempt -> outcome
  local o=$(awk -v s=$1 -v t=$2 -v m=$3 '$1==s && $2==t && $3==m {print $4}' $SIM 2>/dev/null)
  [ -z "$o" ] && { echo done; return; }
  echo "$o" | tr ',' '\n' | sed -n "${4}p" | grep . || echo "$o" | tr ',' '\n' | tail -1
}

# Sets ST, CLS, ROW. Every attempt gets its own --out directory.
run_one() {  # seed task model attempt timeout
  local s=$1 t=$2 m=$3 a=$4 to=$5
  local dir=$OUT/cells/seed$s/$t-$m/attempt$a
  log "START seed$s $t $m attempt$a"
  wait_idle
  local free=$( [ $DRY -eq 1 ] && echo dry || nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  local start=$(date -Is) s0=$(date +%s) cmd=($PY $WT/scripts/validation_v2.py run --task $t --model $m --profile full
                                             --seeds $s --timeout $to --out $dir)
  local score= metric= err= wall
  if [ $DRY -eq 1 ]; then
    say "  [dry-run] (cd $WT && ${cmd[*]})"
    local o=$(if [ -n "$SIM" ]; then sim_outcome $s $t $m $a; else echo done; fi)
    case $o in done) ST=done; CLS=done ;; timeout) ST=timeout; CLS=timeout ;; oom) ST=failed; CLS=oom ;;
               cuda_init) ST=failed; CLS=cuda_init ;; *) ST=failed; CLS=other ;; esac
    wall=$([ $CLS = cuda_init ] && echo 2 || echo 0); err="simulated $o"
  else
    mkdir -p $(dirname $dir)
    (cd $WT && nvidia-smi > $dir.nvidia_smi_before.txt 2>&1; timeout $((to + 600)) "${cmd[@]}" > $dir.stdout 2>&1)
    wall=$(( $(date +%s) - s0 ))
    local res=$(ls -t $dir/*/result.json 2>/dev/null | head -1)
    ST=$(field "$res" "d['status']"); ST=${ST:-no_result}
    metric=$(field "$res" "json.dumps(d.get('metrics'))")
    score=$(field "$res" "d['metrics'][d['metric']['name']]")
    err=$(field "$res" "d.get('error','')[:200].replace('\t',' ').replace('\n',' ')")
    local texts=$( (field "$res" "d.get('error','')"; cat $dir/*/train.log $dir.stdout) 2>/dev/null )
    if [ "$ST" = done ]; then CLS=done
    elif [ "$ST" = timeout ]; then CLS=timeout
    elif [ $wall -le 10 ] && echo "$texts" | grep -qE "$CUDA_RE"; then CLS=cuda_init
    elif echo "$texts" | grep -qE "$OOM_RE"; then CLS=oom
    else CLS=other; fi
  fi
  ROW="$s\t$t\t$m\t$a\t$ST\t$CLS\t$start\t$(date -Is)\t$wall\t$to\t$free\t$score\t$metric\t$dir\t$err"
  echo -e "$ROW" >> $ATTEMPTS
  log "END seed$s $t $m attempt$a status=$ST class=$CLS wall=${wall}s"
}

summarize() { $PY $WT/scripts/nightly_v2_summary.py $OUT && log "SUMMARY_WRITTEN"; }
seed_done() {  # seed
  log "SEED_DONE seed$1 cells=$(awk -F'\t' -v s=$1 'NR > 1 && $1 == s' $STATUS | wc -l)"
  $PY $WT/scripts/nightly_v2_summary.py $OUT $1
}

# ---------------------------------------------------------------- main loop
log "BEGIN nightly_v2 git=$(git -C $WT rev-parse HEAD) branch=$(git -C $WT rev-parse --abbrev-ref HEAD) seeds=[$SEEDS] cells=${#ORDER[@]} deadline=${DEADLINE:-none} dry_run=$DRY"
CONSEC=0 PREV=
for entry in "${ORDER[@]}"; do
  read -r s t m to <<< "$entry"
  [ -n "$PREV" ] && [ "$PREV" != "$s" ] && { seed_done $PREV; summarize; }
  PREV=$s
  if [ -n "$DEADLINE_S" ] && [ $(date +%s) -ge $DEADLINE_S ]; then
    log "DEADLINE reached before seed$s $t $m; no new cell started"; summarize; exit 0
  fi
  a=0 cuda_retries=0 other_retries=0
  while true; do
    a=$((a + 1)); run_one $s $t $m $a $to
    case $CLS in
      done|timeout) break ;;
      cuda_init)
        gpu_check $OUT/cells/seed$s/$t-$m/attempt$a.gpu_check.txt
        if [ $cuda_retries -ge 3 ]; then ST=cuda_init_failed; break; fi
        cuda_retries=$((cuda_retries + 1)); pause=$((30 + RANDOM % 31))
        log "CUDA_INIT_RETRY seed$s $t $m retry$cuda_retries after ${pause}s"
        [ $DRY -eq 1 ] && say "  [dry-run] sleep $pause" || sleep $pause ;;
      *)
        if [ $other_retries -ge 1 ]; then break; fi
        other_retries=1
        if [ $CLS = oom ]; then log "OOM_RETRY seed$s $t $m after 60s"; [ $DRY -eq 1 ] && say "  [dry-run] sleep 60" || sleep 60
        else log "RETRY seed$s $t $m class=$CLS"; fi ;;
    esac
  done
  echo -e "$(echo -e "$ROW" | awk -F'\t' -v OFS='\t' -v st=$ST '{$5=st; print}')" >> $STATUS
  log "CELL_DONE seed$s $t $m status=$ST attempts=$a"
  if [ "$ST" = cuda_init_failed ]; then CONSEC=$((CONSEC + 1)); else CONSEC=0; fi
  if [ $CONSEC -ge 3 ]; then
    { echo "ABORT $(date -Is): 3 consecutive cuda_init_failed cells, last seed$s $t $m"
      echo "--- last error"; echo -e "$ROW" | cut -f15
      [ $DRY -eq 1 ] && echo "(dry-run: nvidia-smi not called)" || { echo "--- nvidia-smi"; nvidia-smi 2>&1; }; } > $OUT/ABORT
    log "ABORT consecutive_cuda_init_failed=3 at seed$s $t $m -> $OUT/ABORT"; summarize; exit 3
  fi
done
[ -n "$PREV" ] && seed_done $PREV
log "ALL_DONE"
summarize
