#!/usr/bin/env bash
# Autoresearch benchmark: C decode tok/s (higher is better).
# Usage: bash autoresearch.sh
# Env overrides: N, CCD, CPU, WARM, STEADY, MODEL, BIN
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="${BIN:-c_infer/ll3m_c}"
MODEL="${MODEL:-checkpoints/v76_dense_c/best.ll3m}"
N="${N:-256}"
CCD="${CCD:-1}"
CPU="${CPU:-8}"
WARM="${WARM:-24}"
STEADY="${STEADY:-64}"
TOKEN_IDS="${TOKEN_IDS:-12,34,56,78,90,100,110,120,130,140,150,160}"

if [[ ! -x "$BIN" ]]; then
  echo "missing binary $BIN" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing model $MODEL" >&2
  exit 2
fi

ulimit -l unlimited 2>/dev/null || true

LOG="$(mktemp)"
set +e
taskset -c "$CPU" "$BIN" \
  -m "$MODEL" \
  --token-ids "$TOKEN_IDS" \
  -n "$N" \
  --greedy \
  --residency-hard \
  --ccd "$CCD" \
  --warm "$WARM" \
  --steady-warm "$STEADY" \
  >"$LOG" 2>&1
RC=$?
set -e

# Optional fields must not abort the script (no soft_fail line when mask=0).
TOK=$(rg -o '[0-9]+\.[0-9]+ tok/s' "$LOG" 2>/dev/null | head -1 | awk '{print $1}') || true
DRAM=$(rg -o 'DRAM share [0-9.]+%' "$LOG" 2>/dev/null | head -1 | awk '{print $3}' | tr -d '%') || true
LABEL=$(rg -o 'L3 residency \(fill source\): [A-Z]+' "$LOG" 2>/dev/null | head -1 | awk '{print $NF}') || true
FIFO=$(rg -o 'SCHED_FIFO:[[:space:]]+[A-Z]+' "$LOG" 2>/dev/null | head -1 | awk '{print $2}') || true
MASK=$(rg -o 'soft_fail_mask: 0x[0-9a-fA-F]+' "$LOG" 2>/dev/null | head -1 | awk '{print $2}') || true
WS=$(rg -o 'infer WS:[[:space:]]+[0-9.]+ MiB' "$LOG" 2>/dev/null | head -1 | awk '{print $3}') || true
FITS=$(rg -o 'fits one CCD L3: [A-Z]+' "$LOG" 2>/dev/null | head -1 | awk '{print $NF}') || true

echo "=== autoresearch benchmark ==="
echo "rc=$RC"
echo "tok_s=${TOK:-NA}"
echo "dram_pct=${DRAM:-NA}"
echo "residency_label=${LABEL:-NA}"
echo "fifo=${FIFO:-NA}"
echo "soft_fail_mask=${MASK:-none}"
echo "infer_ws_mib=${WS:-NA}"
echo "fits_ccd=${FITS:-NA}"
echo "--- key log lines ---"
rg -n "tok/s|DRAM|SCHED_FIFO|mlockall|CPU bind|infer WS|fits one|RESIDENCY|L3 residency" "$LOG" 2>/dev/null || true
echo "--- end ---"
echo "full_log=$LOG"

if [[ -z "${TOK:-}" ]]; then
  exit 3
fi
# Always 0 if we got a metric (even if soft_fail bits set); loop decides on tok_s.
exit 0
