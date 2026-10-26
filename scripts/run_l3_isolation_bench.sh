#!/usr/bin/env bash
# Best-effort dual-CCD residency + throughput bench for LL3M C decode.
#
# Ryzen 9 9950X (this host):
#   CCD0 L3: CPUs 0-7,16-23
#   CCD1 L3: CPUs 8-15,24-31
#
# Without boot isolcpus we cannot fully empty a CCD of IRQs/kernel work.
# This script still:
#   1) pins the decode process to one physical core on the quiet CCD
#   2) best-effort moves *this shell's* cpuset / affinity away from that CCD
#   3) runs soft + hard residency with PMU and records tok/s + DRAM%
#
# Usage:
#   bash scripts/run_l3_isolation_bench.sh
#   MODEL=checkpoints/v76_dense_c/best.ll3m QUIET_CCD=1 N=128 bash scripts/run_l3_isolation_bench.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-checkpoints/v76_dense_c/best.ll3m}"
BIN="${BIN:-c_infer/ll3m_c}"
OUT_DIR="${OUT_DIR:-checkpoints/compare/l3_isolation_bench}"
N="${N:-128}"
WARM="${WARM:-24}"
STEADY="${STEADY:-64}"
# Quiet CCD for decode (1 = second 32 MiB slice on 9950X)
QUIET_CCD="${QUIET_CCD:-1}"
# Physical core index within that CCD for single-core hard runs (0..7)
CORE_IN_CCD="${CORE_IN_CCD:-0}"
# Token prefill (C has no BPE)
TOKEN_IDS="${TOKEN_IDS:-12,34,56,78,90,100,110,120,130,140,150,160,170,180,190,200}"

if [[ ! -x "$BIN" ]]; then
  make -C c_infer -j
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing MODEL=$MODEL" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="$OUT_DIR/summary_${STAMP}.md"
LOG_DIR="$OUT_DIR/logs_${STAMP}"
mkdir -p "$LOG_DIR"

# Map QUIET_CCD + CORE_IN_CCD → one physical CPU id (prefer non-SMT first half)
# CCD0 physical cores: 0-7  (SMT 16-23)
# CCD1 physical cores: 8-15 (SMT 24-31)
if [[ "$QUIET_CCD" -eq 0 ]]; then
  DECODE_CPU=$((CORE_IN_CCD))
  NOISE_MASK="8-15,24-31"   # keep noise on CCD1 when decode on CCD0
  DECODE_MASK="$DECODE_CPU"
  QUIET_LIST="0-7,16-23"
else
  DECODE_CPU=$((8 + CORE_IN_CCD))
  NOISE_MASK="0-7,16-23"
  DECODE_MASK="$DECODE_CPU"
  QUIET_LIST="8-15,24-31"
fi

{
  echo "# L3 isolation bench"
  echo
  echo "- stamp: \`$STAMP\`"
  echo "- host: \`$(uname -n)\` · \`$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\`"
  echo "- model: \`$MODEL\`"
  echo "- quiet CCD: **$QUIET_CCD** (CPUs $QUIET_LIST)"
  echo "- decode CPU (single-core hard): **$DECODE_CPU**"
  echo "- noise affinity hint: \`$NOISE_MASK\`"
  echo "- isolcpus: \`$(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo none)\`"
  echo "- n_tokens timed: $N · warm=$WARM · steady-warm=$STEADY"
  echo
} | tee "$SUMMARY"

# Best-effort: pin *this* script / leftover work to noise CCD
if command -v taskset >/dev/null 2>&1; then
  taskset -cp "$NOISE_MASK" $$ >/dev/null 2>&1 || true
fi

run_one() {
  local tag="$1"
  shift
  local log="$LOG_DIR/${tag}.txt"
  echo "## $tag" | tee -a "$SUMMARY"
  echo '```' >>"$SUMMARY"
  # Decode process affinity: for hard/single we also pass --ccd; taskset enforces CPU
  set +e
  if [[ "$tag" == *hard* || "$tag" == *single* ]]; then
    taskset -c "$DECODE_MASK" "$BIN" "$@" >"$log" 2>&1
  else
    # soft: whole quiet CCD allowed
    taskset -c "$QUIET_LIST" "$BIN" "$@" >"$log" 2>&1
  fi
  local rc=$?
  set -e
  # Extract key lines
  rg -n "tok/s|L3 residency|DRAM|infer WS|fits one|SCHED_FIFO|isolcpus|mlockall|CCD bind|L3_RESIDENT|RESIDENCY_HARD|weights:" "$log" \
    | tee -a "$SUMMARY" || true
  echo "exit_rc=$rc log=$log" | tee -a "$SUMMARY"
  echo '```' >>"$SUMMARY"
  echo | tee -a "$SUMMARY"

  # machine-readable crumbs
  local toks
  toks=$(rg -o '[0-9.]+ tok/s' "$log" | head -1 | awk '{print $1}')
  local dram
  dram=$(rg -o 'DRAM share [0-9.]+%' "$log" | head -1 | awk '{print $3}' | tr -d '%')
  local label
  label=$(rg -o 'L3 residency \(fill source\): [A-Z]+' "$log" | head -1 | awk '{print $NF}')
  printf '%s\ttok_s=%s\tdram_pct=%s\tlabel=%s\trc=%s\n' \
    "$tag" "${toks:-?}" "${dram:-?}" "${label:-?}" "$rc" >>"$LOG_DIR/metrics.tsv"
}

COMMON=( -m "$MODEL" --token-ids "$TOKEN_IDS" -n "$N" --greedy )

echo "metrics_tsv=$LOG_DIR/metrics.tsv" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"

# 1) Soft residency on quiet CCD (multi-core allowed on that CCD)
run_one "soft_quiet_ccd${QUIET_CCD}" \
  "${COMMON[@]}" --l3-resident --ccd "$QUIET_CCD" \
  --warm "$WARM" --steady-warm "$STEADY"

# 2) Hard residency, engine ccd + single core (taskset to DECODE_CPU)
run_one "hard_quiet_cpu${DECODE_CPU}" \
  "${COMMON[@]}" --residency-hard --ccd "$QUIET_CCD" \
  --warm "$WARM" --steady-warm "$STEADY"

# 3) Soft on the *busy* CCD for contrast (noise CCD)
BUSY_CCD=$((1 - QUIET_CCD))
run_one "soft_busy_ccd${BUSY_CCD}" \
  "${COMMON[@]}" --l3-resident --ccd "$BUSY_CCD" \
  --warm "$WARM" --steady-warm "$STEADY"

# 4) Longer hard run for stabler tok/s
run_one "hard_quiet_cpu${DECODE_CPU}_n$((N*2))" \
  -m "$MODEL" --token-ids "$TOKEN_IDS" -n $((N * 2)) --greedy \
  --residency-hard --ccd "$QUIET_CCD" \
  --warm "$WARM" --steady-warm "$STEADY"

{
  echo "## Metrics table"
  echo
  echo '```'
  column -t "$LOG_DIR/metrics.tsv" 2>/dev/null || cat "$LOG_DIR/metrics.tsv"
  echo '```'
  echo
  echo "## Interpretation notes"
  echo
  echo "- Without boot \`isolcpus\`/\`irqaffinity\`, DRAM% may stay WEAK even on a quiet CCD."
  echo "- Compare soft_quiet vs soft_busy and hard_quiet tok/s + DRAM%."
  echo "- For true empty-CCD proof, reboot with e.g.:"
  echo "  \`isolcpus=${DECODE_CPU} nohz_full=${DECODE_CPU} rcu_nocbs=${DECODE_CPU} irqaffinity=${NOISE_MASK}\`"
  echo "  then re-run with \`--residency-hard --ccd ${QUIET_CCD}\`."
  echo
  echo "Artifacts: \`$LOG_DIR\` · summary \`$SUMMARY\`"
} | tee -a "$SUMMARY"

echo
echo "DONE summary=$SUMMARY"
