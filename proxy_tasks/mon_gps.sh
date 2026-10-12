#!/usr/bin/env bash
# ZINC GraphGPS 실험 진행 모니터
cd "$(dirname "$0")"
L=logs/zinc_gps
TOTAL=500
printf '=== ZINC GraphGPS  %s ===\n' "$(date '+%H:%M:%S')"

if pgrep -f "zinc.train --model gps" >/dev/null; then
  printf 'STATUS : RUNNING  (pid %s)\n' "$(pgrep -f 'zinc.train --model gps' | tr '\n' ' ')"
else
  grep -q "ALL DONE" "$L/driver.log" 2>/dev/null \
    && printf 'STATUS : ALL DONE\n' || printf 'STATUS : NOT RUNNING (미시작 또는 중단)\n'
fi

printf 'GPU    : %s\n' "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
  --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo
printf '%-6s %-7s %-9s %-9s %-9s %-8s %s\n' SEED EPOCH PROGRESS VAL_MAE ELAPSED S/EP ETA
for s in 0 1 2; do
  f="$L/gps_s${s}_e500.log"
  [ -f "$f" ] || { printf '%-6s %s\n' "$s" "(로그 없음)"; continue; }
  line=$(grep -a 'epoch=' "$f" | tail -1)
  if [ -z "$line" ]; then printf '%-6s %s\n' "$s" "(워밍업)"; continue; fi
  ep=$(sed -n 's/.*epoch=\([0-9]*\).*/\1/p' <<<"$line")
  vm=$(sed -n 's/.*val_mae=\([0-9.]*\).*/\1/p' <<<"$line")
  el=$(sed -n 's/.*elapsed=\([0-9.]*\)s.*/\1/p' <<<"$line")
  done_mark=$(grep -ac 'final test_mae' "$f")
  awk -v s="$s" -v ep="$ep" -v vm="$vm" -v el="$el" -v tot="$TOTAL" -v dm="$done_mark" 'BEGIN{
    spe = (ep>0)? el/ep : 0
    eta = (dm>0)? 0 : spe*(tot-ep)
    printf "%-6s %-7s %-9s %-9s %-9s %-8s %s\n", s, ep,
      sprintf("%.0f%%", 100*ep/tot), vm, sprintf("%.0fs", el),
      sprintf("%.2f", spe),
      (dm>0)? "DONE" : sprintf("%.0fm", eta/60)
  }'
done
echo
echo '--- 완료된 결과 ---'
for s in 0 1 2; do
  f="$L/gps_s${s}_e500.log"
  [ -f "$f" ] && grep -aH -e 'final test_mae' -e 'e200 test_mae' "$f" | sed 's#.*/##'
done
echo
echo '--- driver ---'
tail -4 "$L/driver.log" 2>/dev/null
