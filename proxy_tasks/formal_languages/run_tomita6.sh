#!/usr/bin/env bash
set -u
REPO=${REPO:-/workspace/tfl}; EXP_DIR=${EXP_DIR:-/workspace/exp/2026-10-12}
RUNNER=${RUNNER:-$EXP_DIR/runner.py}; P=${P:-8}; MODE=${1:-full}
export EXP_DIR; mkdir -p "$EXP_DIR/logs"; cd "$REPO" || exit 1
COMMON="-mode train -dataset Tomita-6 -opt rmsprop -batch_size 32 -gpu 0"
start_gpu_poll() {
  if [ ! -f "$EXP_DIR/gpu_poll.pid" ] || ! kill -0 "$(cat "$EXP_DIR/gpu_poll.pid")" 2>/dev/null; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -l 5 >> "$EXP_DIR/gpu.csv" &
    echo $! > "$EXP_DIR/gpu_poll.pid"; fi; }
capture_env() {
  [ -f "$EXP_DIR/env.json" ] && return
  { echo "tfl_commit: $(git -C "$REPO" rev-parse HEAD)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; pip freeze; } > "$EXP_DIR/env.txt"; }
write_commands() {
  local f="$EXP_DIR/tomita6_cmds.txt"; : > "$f"
  for d in 16 32; do for L in 1 2; do
    echo "SAN -pos_encode -d_model $d -depth $L -heads 4 -lr 1e-3|E2T6-SAN-abs-d$d-L$L" >> "$f"
  done; done
  for h in 16 32; do for lr in 1e-2 1e-3; do
    echo "RNN -cell_type LSTM -hidden_size $h -depth 1 -lr $lr|E2T6-LSTM-h$h-lr$lr" >> "$f"
  done; done; echo "$f"; }
run_one() {
  local model_args=${1%%|*} name=${1##*|}; local arch=${model_args%% *} rest=${model_args#* }
  N_CONCURRENT=$P EXP_ID=E2-T6 python "$RUNNER" $COMMON -model_type $arch $rest -epochs 100 -run_name "$name" \
    > "$EXP_DIR/logs/$name.log" 2>&1 && echo "[$(date +%H:%M:%S)] done   $name" || echo "[$(date +%H:%M:%S)] FAILED $name"; }
export -f run_one; export COMMON P RUNNER EXP_DIR
case "$MODE" in
  pilot)
    start_gpu_poll; capture_env
    N_CONCURRENT=1 EXP_ID=E1-pilot python "$RUNNER" $COMMON -model_type SAN -pos_encode -d_model 32 -depth 2 -heads 4 \
      -lr 1e-3 -epochs 2 -run_name E2T6pilot-SAN-d32-L2 > "$EXP_DIR/logs/E2T6pilot.log" 2>&1 || { echo "pilot 실패 → tail -40 $EXP_DIR/logs/E2T6pilot.log"; exit 1; }
    python -c "
import json
r=[json.loads(l) for l in open('$EXP_DIR/runs.jsonl') if 'E2T6pilot' in l][-1]
e=(r['t_train_s']+r['t_val_s'])/2
print(f\"epoch당 {e:.1f}s (train {r['t_train_s']/2:.1f} + val {r['t_val_s']/2:.1f}), setup {r['t_setup_s']:.1f}s\")
print(f'Transformer 1 run(100 epoch, 단독) ≈ {e*100/60:.0f}분')" ;;
  full)
    start_gpu_poll; capture_env; cmds=$(write_commands)
    echo "[$(date +%H:%M:%S)] $(wc -l < "$cmds") run 시작, 동시 $P개"
    tr '\n' '\0' < "$cmds" | xargs -0 -P "$P" -I{} bash -c 'run_one "$@"' _ {}
    echo "[$(date +%H:%M:%S)] 전부 종료" ;;
  summary) python "$EXP_DIR/summarize.py" "$EXP_DIR/runs.jsonl" E2-T6 ;;
esac
