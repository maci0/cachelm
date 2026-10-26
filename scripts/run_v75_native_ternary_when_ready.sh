#!/usr/bin/env bash
# Wait for GPU train free → native ternary (research recipe) → export + compares
# → optional tool SFT (RUN_TOOL_SFT=1).
#
# Research pipeline: docs/NATIVE_TERNARY_PIPELINE.md
#
#   nohup bash scripts/run_v75_native_ternary_when_ready.sh \
#     >> checkpoints/v75_native_ternary/waiter.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-checkpoints/v75_native_ternary}"
WAIT_PID="${WAIT_PID:-}"
RUN_TOOL_SFT="${RUN_TOOL_SFT:-1}"

if [[ -z "$WAIT_PID" && -f checkpoints/v75_ternary/train.pid ]]; then
  WAIT_PID=$(cat checkpoints/v75_ternary/train.pid 2>/dev/null || true)
fi
if [[ -z "$WAIT_PID" ]]; then
  WAIT_PID=$(pgrep -f "ll3m.train" 2>/dev/null | head -1 || true)
fi

mkdir -p "$OUT"
LOG="$OUT/waiter.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

if [[ -n "${WAIT_PID}" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  log "waiting for train pid $WAIT_PID before native ternary..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    if [[ -f checkpoints/v75_ternary/progress.json ]]; then
      step=$("$PY" -c "import json; print(json.load(open('checkpoints/v75_ternary/progress.json')).get('step','?'))" 2>/dev/null || echo "?")
      log "  still running (watched) step=$step"
    else
      log "  still running pid=$WAIT_PID"
    fi
    sleep 60
  done
  log "watched process exited"
  sleep 15
  for _ in $(seq 1 40); do
    if pgrep -f "ll3m.train|compare_fp32_vs_qat|export_weights" >/dev/null 2>&1; then
      log "  post-train job still busy; sleep 30s"
      sleep 30
    else
      break
    fi
  done
else
  log "no live train to wait on; starting native ternary now"
fi

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "ERROR: no GPU torch — abort native launch"
  exit 1
fi

log "=== launch native ternary (research recipe) ==="
# Not exec: we need post-steps after train
nohup bash "$ROOT/scripts/train_v75_native_ternary.sh" \
  >>"$OUT/train.log" 2>&1 &
TPID=$!
echo "$TPID" >"$OUT/train.pid"
log "started native ternary pid=$TPID  log=$OUT/train.log"
wait "$TPID"
RC=$?
log "native ternary exited rc=$RC"
if [[ $RC -ne 0 ]]; then
  exit "$RC"
fi

if [[ -f "$OUT/best.pt" ]]; then
  log "export L3T2 → $OUT/best.ll3m"
  "$PY" scripts/export_weights.py \
    --checkpoint "$OUT/best.pt" \
    --out "$OUT/best.ll3m" \
    --quant ternary \
    --tokenizer data/tokenizer.json \
    >>"$LOG" 2>&1 || log "export failed"
fi

if [[ -f checkpoints/v75/best.pt && -f "$OUT/best.pt" ]]; then
  log "compare: fp32 v75 vs native ternary"
  "$PY" scripts/compare_fp32_vs_qat.py \
    --baseline checkpoints/v75/best.pt \
    --qat "$OUT/best.pt" \
    --out "$OUT/quality_compare_vs_fp32.json" \
    >>"$LOG" 2>&1 || log "compare vs fp32 failed"
fi
if [[ -f checkpoints/v75_ternary/best.pt && -f "$OUT/best.pt" ]]; then
  log "compare: posthoc QAT vs native ternary"
  "$PY" scripts/compare_fp32_vs_qat.py \
    --baseline checkpoints/v75_ternary/best.pt \
    --qat "$OUT/best.pt" \
    --out "$OUT/quality_compare_vs_posthoc.json" \
    >>"$LOG" 2>&1 || log "compare vs posthoc failed"
fi

if [[ "$RUN_TOOL_SFT" == "1" && -f "$OUT/best.pt" ]]; then
  log "=== tool SFT (When2Call mix, keep ternary grid) ==="
  CKPT="$OUT/best.pt" OUT="${OUT}_tool" \
    bash "$ROOT/scripts/train_tool_sft.sh" >>"$LOG" 2>&1 \
    || log "tool SFT failed (non-fatal for base native weights)"
fi

log "done. native out=$OUT  pipeline=docs/NATIVE_TERNARY_PIPELINE.md"
