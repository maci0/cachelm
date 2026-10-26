#!/usr/bin/env bash
# Stop dense ladder on this host (spark1). Safe if nothing running.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
screen -S ll3m-dense -X quit 2>/dev/null || true
# PIDs of train / prepare / download (python -u scripts/… or -m ll3m.train)
mapfile -t PIDS < <(ps -u "$USER" -o pid=,args= | awk '
  /[l]l3m\.train/ {print $1}
  /[p]repare_base_english/ {print $1}
  /[d]ownload_base_english/ {print $1}
  /[r]un_spark1_dense_pipeline/ {print $1}
')
for pid in "${PIDS[@]:-}"; do
  kill "$pid" 2>/dev/null || true
done
sleep 2
for pid in "${PIDS[@]:-}"; do
  kill -9 "$pid" 2>/dev/null || true
done
echo "remaining:"
ps -u "$USER" -o pid=,args= | awk '/[l]l3m\.train|[p]repare_base|[d]ownload_base|[l]l3m-dense/ {print}' || true
screen -ls 2>/dev/null || true
# Drop partial dense tree so next launch rebuilds
rm -rf checkpoints/v76_dense_c checkpoints/v76_dense_qa checkpoints/v76_dense_tool data/base_en_dense
mkdir -p checkpoints/dense_pipeline
echo "[$(date -Iseconds)] KILLED; denser rebuild + full QAT" >>checkpoints/dense_pipeline/pipeline.log
echo KILLED_OK
