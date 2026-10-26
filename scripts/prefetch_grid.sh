#!/usr/bin/env bash
# R2: software-prefetch knob grid for C decode (docs/READAHEAD_DESIGN.md).
# Builds each config into a temp binary, runs autoresearch-style hard+FIFO bench.
#
# Usage:
#   bash scripts/prefetch_grid.sh
#   AHEADS="16 32 64" HINTS="0 1" bash scripts/prefetch_grid.sh
#
# Keep rule (same as autoresearch): ≥1% tok/s over baseline (AHEAD=32 HINT=0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-checkpoints/v76_dense_c/best.ll3m}"
N="${N:-256}"
CCD="${CCD:-1}"
CPU="${CPU:-8}"
WARM="${WARM:-24}"
STEADY="${STEADY:-64}"
TOKEN_IDS="${TOKEN_IDS:-12,34,56,78,90,100,110,120,130,140,150,160}"
AHEADS="${AHEADS:-0 16 32 64 128}"
HINTS="${HINTS:-0 1}"  # 0=NTA 1=T1; T0 often pollutes L1d
NEXT_LAYER="${NEXT_LAYER:-1}"
OUTDIR="${OUTDIR:-checkpoints/compare/prefetch_grid}"
mkdir -p "$OUTDIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
CSV="$OUTDIR/grid_${TS}.csv"
JSONL="$OUTDIR/grid_${TS}.jsonl"

if [[ ! -f "$MODEL" ]]; then
  echo "missing $MODEL" >&2
  exit 2
fi

ulimit -l unlimited 2>/dev/null || true

echo "ahead,hint,next_layer,tok_s,ws_mib,fits,dram_pct,fifo,rc" | tee "$CSV"
: >"$JSONL"

run_one() {
  local ahead="$1" hint="$2" nlayer="$3"
  local tag="a${ahead}_h${hint}_nl${nlayer}"
  echo "=== build $tag ===" >&2
  make -C c_infer -j "$(nproc)" BIN="ll3m_c.prefetch_${tag}" \
    CFLAGS="-O3 -march=native -mtune=native -mavx512f -mavx512vnni -mfma \
      -ffast-math -fno-math-errno -fno-trapping-math \
      -flto=auto -ffunction-sections -fdata-sections \
      -falign-functions=64 -falign-loops=32 -falign-jumps=32 \
      -fvisibility=hidden -fno-semantic-interposition \
      -std=c11 -Wall -Wextra -pthread \
      -DCLM_QUANT=2 -DCLM_EMBED_QUANT=1 \
      -DCLM_W_PREFETCH_AHEAD=${ahead} \
      -DCLM_W_PREFETCH_HINT=${hint} \
      -DCLM_NEXT_LAYER_PREFETCH=${nlayer}" \
    >/tmp/prefetch_build_${tag}.log 2>&1 || {
      echo "$ahead,$hint,$nlayer,BUILD_FAIL,,,,,1" | tee -a "$CSV"
      return 0
    }
  if command -v setcap >/dev/null 2>&1; then
    sudo -n setcap cap_sys_nice+ep "c_infer/ll3m_c.prefetch_${tag}" 2>/dev/null || true
  fi

  local log
  log="$(mktemp)"
  set +e
  taskset -c "$CPU" "./c_infer/ll3m_c.prefetch_${tag}" \
    -m "$MODEL" \
    --token-ids "$TOKEN_IDS" \
    -n "$N" \
    --greedy \
    --residency-hard \
    --ccd "$CCD" \
    --warm "$WARM" \
    --steady-warm "$STEADY" \
    >"$log" 2>&1
  local rc=$?
  set -e

  local tok dram fifo ws fits
  tok=$(rg -o '[0-9]+\.[0-9]+ tok/s' "$log" 2>/dev/null | head -1 | awk '{print $1}') || true
  dram=$(rg -o 'DRAM share [0-9.]+%' "$log" 2>/dev/null | head -1 | awk '{print $3}' | tr -d '%') || true
  fifo=$(rg -o 'SCHED_FIFO:[[:space:]]+[A-Z]+' "$log" 2>/dev/null | head -1 | awk '{print $2}') || true
  ws=$(rg -o 'infer WS:[[:space:]]+[0-9.]+ MiB' "$log" 2>/dev/null | head -1 | awk '{print $3}') || true
  fits=$(rg -o 'fits one CCD L3: [A-Z]+' "$log" 2>/dev/null | head -1 | awk '{print $NF}') || true

  echo "${ahead},${hint},${nlayer},${tok:-},${ws:-},${fits:-},${dram:-},${fifo:-},${rc}" | tee -a "$CSV"
  printf '{"ahead":%s,"hint":%s,"next_layer":%s,"tok_s":%s,"ws_mib":%s,"fits":"%s","dram_pct":%s,"fifo":"%s","rc":%s,"log":"%s"}\n' \
    "$ahead" "$hint" "$nlayer" \
    "${tok:-null}" "${ws:-null}" "${fits:-}" "${dram:-null}" "${fifo:-}" "$rc" "$log" \
    >>"$JSONL"
}

for ahead in $AHEADS; do
  for hint in $HINTS; do
    for nl in $NEXT_LAYER; do
      run_one "$ahead" "$hint" "$nl"
    done
  done
done

make -C c_infer -j "$(nproc)" >/tmp/prefetch_rebuild_default.log 2>&1 || true
if command -v setcap >/dev/null 2>&1; then
  sudo -n setcap cap_sys_nice+ep c_infer/ll3m_c 2>/dev/null || true
fi

echo "wrote $CSV"
echo "wrote $JSONL"
echo "=== rank (tok/s desc) ==="
tail -n +2 "$CSV" | sort -t, -k4 -nr | head -20
