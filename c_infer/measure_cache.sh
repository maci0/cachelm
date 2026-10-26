#!/usr/bin/env bash
# Prove L3 residency of bare-metal ll3m inference (Linux + perf).
#
# Goal: after pin+warm, DRAM should be a tiny share of L2 fills; infer WS ≪ CCD L3.
#
# Usage:
#   ./measure_cache.sh [model.clm] [n_tokens]
#   ./measure_cache.sh ../checkpoints/best_L6.clm 200
#
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-../checkpoints/best_L6.clm}"
NTOK="${2:-200}"
BIN=./ll3m_c
# Fail if DRAM share of L2 fills exceeds this percent (after warm path).
DRAM_MAX_PCT="${DRAM_MAX_PCT:-5}"

if [[ ! -x "$BIN" ]]; then
  make -j
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi
if ! command -v perf >/dev/null; then
  echo "perf not found (install linux-tools / perf)" >&2
  exit 1
fi

OUT=$(mktemp)
trap 'rm -f "$OUT" "$OUT.stdout"' EXIT

EVENTS=(
  cycles
  instructions
  cache-references
  cache-misses
  L1-dcache-loads
  L1-dcache-load-misses
  l2_cache_req_stat.dc_access_in_l2
  l2_cache_req_stat.dc_hit_in_l2
  l2_cache_req_stat.ic_dc_miss_in_l2
  l2_fill_rsp_src.dram_io_near
  l2_fill_rsp_src.dram_io_far
  l2_fill_rsp_src.local_ccx
  l2_fill_rsp_src.near_cache
  l2_fill_rsp_src.far_cache
)

IFS=,
EVENT_STR="${EVENTS[*]}"
unset IFS

echo "=== perf cache stats: L3-resident path ==="
echo "model: $MODEL  tokens: $NTOK  DRAM_MAX_PCT=${DRAM_MAX_PCT}%"
echo

# --residency-hard: mlockall, 1-core CCD pin, FIFO, hierarchical L1→L2→L3 touch
FLAGS=(--greedy --residency-hard --warm 16)
if ! perf stat -e "$EVENT_STR" -- \
  "$BIN" -m "$MODEL" -p $'ROMEO:\n' -n "$NTOK" "${FLAGS[@]}" \
  >"$OUT.stdout" 2>"$OUT"; then
  echo "(retrying with portable counters only)"
  perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses -- \
    "$BIN" -m "$MODEL" -p $'ROMEO:\n' -n "$NTOK" "${FLAGS[@]}" \
    >"$OUT.stdout" 2>"$OUT" || true
fi

grep -E 'weights:|activations:|infer WS:|fits |mlock|L3-resident|residency-hard|tok/s|config:|GEMV|L3_RESIDENT|RESIDENCY_HARD|L1d pad|L3 fill|local L3|DRAM' "$OUT.stdout" || true
echo
echo "--- raw perf ---"
grep -E 'cycles|instructions|cache-|L1-|l2_|seconds time|not supported|not counted' "$OUT" || cat "$OUT"
echo
echo "--- rates ---"

get_count() {
  local name="$1"
  local line
  # perf may print "name" or "name:u"
  line=$(grep -E "[[:space:]]${name}(:u)?([[:space:]]|$)" "$OUT" | head -1 || true)
  if [[ -z "$line" ]] || echo "$line" | grep -q 'not supported\|not counted'; then
    echo ""
    return
  fi
  echo "$line" | awk '{gsub(/,/, "", $1); print $1}'
}

num() { [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ ]]; }

cref=$(get_count "cache-references")
cmiss=$(get_count "cache-misses")
l1ld=$(get_count "L1-dcache-loads")
l1miss=$(get_count "L1-dcache-load-misses")
l2acc=$(get_count "l2_cache_req_stat.dc_access_in_l2")
l2hit=$(get_count "l2_cache_req_stat.dc_hit_in_l2")
l2miss=$(get_count "l2_cache_req_stat.ic_dc_miss_in_l2")
dram_n=$(get_count "l2_fill_rsp_src.dram_io_near")
dram_f=$(get_count "l2_fill_rsp_src.dram_io_far")
l3loc=$(get_count "l2_fill_rsp_src.local_ccx")
near_c=$(get_count "l2_fill_rsp_src.near_cache")
far_c=$(get_count "l2_fill_rsp_src.far_cache")
ins=$(get_count "instructions")
cyc=$(get_count "cycles")

pct() {
  awk -v a="$1" -v b="$2" 'BEGIN{ if(b+0==0) print "n/a"; else printf "%.2f", 100.0*a/b }'
}

if num "$cref" && num "$cmiss"; then
  echo "LLC-ish miss rate (cache-misses/cache-references): $(pct "$cmiss" "$cref")%"
fi
if num "$l1ld" && num "$l1miss"; then
  echo "L1d load miss rate: $(pct "$l1miss" "$l1ld")%"
fi
if num "$l2acc" && num "$l2hit"; then
  echo "L2 DC hit rate: $(pct "$l2hit" "$l2acc")%"
fi

fills=0
for v in "$dram_n" "$dram_f" "$l3loc" "$near_c" "$far_c"; do
  if num "$v"; then fills=$((fills + v)); fi
done
dram_share="n/a"
if [[ "$fills" -gt 0 ]]; then
  echo "L2 fill sources:"
  num "$l3loc" && echo "  local L3:     $l3loc  $(pct "$l3loc" "$fills")%"
  num "$near_c" && echo "  near cache:  $near_c  $(pct "$near_c" "$fills")%"
  num "$far_c" && echo "  far cache:   $far_c  $(pct "$far_c" "$fills")%"
  num "$dram_n" && echo "  DRAM near:   $dram_n  $(pct "$dram_n" "$fills")%"
  num "$dram_f" && echo "  DRAM far:    $dram_f  $(pct "$dram_f" "$fills")%"
  if num "$dram_n" && num "$dram_f"; then
    dram=$((dram_n + dram_f))
    dram_share=$(pct "$dram" "$fills")
    echo "  DRAM share of fills: ${dram_share}%  (target: ~0 for L3-resident hot path)"
  fi
fi
if num "$ins" && num "$cyc"; then
  awk -v i="$ins" -v c="$cyc" 'BEGIN{printf "IPC: %.2f\n", i/c}'
fi

echo
echo "=== L3 residency gate ==="
ws_line=$(grep 'infer WS:' "$OUT.stdout" || true)
fits=$(grep 'fits 32 MiB CCD L3: YES' "$OUT.stdout" || true)
echo "  $ws_line"
if [[ -n "$fits" ]]; then
  echo "  WS fits one CCD L3: YES"
else
  echo "  WS fits one CCD L3: NO"
fi

gate_ok=1
if [[ -z "$fits" ]]; then
  gate_ok=0
fi
if [[ "$dram_share" != "n/a" ]]; then
  # compare as float
  awk -v s="$dram_share" -v m="$DRAM_MAX_PCT" 'BEGIN{exit !(s+0 <= m+0)}' || gate_ok=0
  echo "  DRAM share ${dram_share}%  (max allowed ${DRAM_MAX_PCT}%)"
fi

if [[ "$gate_ok" -eq 1 ]]; then
  echo "L3_RESIDENCY_GATE: PASS"
  exit 0
else
  echo "L3_RESIDENCY_GATE: FAIL  (raise DRAM_MAX_PCT, shrink model, or check pin/warm)"
  exit 1
fi
