#!/usr/bin/env bash
# Poll spark2 native ternary train; when finished, rsync into a SEPARATE local
# directory so a concurrent local native run is never overwritten.
#
#   Remote path:  ~/Desktop/cachelm/checkpoints/v75_native_ternary/   (spark2)
#   Local dest:   checkpoints/v75_native_ternary_spark2/              (default)
#   Local ROCm:   checkpoints/v75_native_ternary/                     (untouched)
#
#   nohup bash scripts/pull_spark2_native_when_ready.sh \
#     >> checkpoints/v75_native_ternary_spark2/pull_waiter.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

KEY="${NVSYNC_KEY:-$HOME/.config/NVIDIA/Sync/config/nvsync.key}"
HOST="${SPARK_HOST:-maci@192.168.0.212}"
REMOTE_DIR="${REMOTE_DIR:-Desktop/cachelm}"
# Path on spark2 (train out-dir there)
REMOTE_OUT="${REMOTE_OUT:-checkpoints/v75_native_ternary}"
# Local destination — MUST differ from local ROCm native out-dir
OUT_LOCAL="${OUT_LOCAL:-checkpoints/v75_native_ternary_spark2}"
LOCAL_NATIVE="${LOCAL_NATIVE:-checkpoints/v75_native_ternary}"

SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=20 "$HOST")

if [[ "$OUT_LOCAL" == "$LOCAL_NATIVE" ]]; then
  echo "ERROR: OUT_LOCAL must not equal local native dir ($LOCAL_NATIVE)" >&2
  echo "  use default checkpoints/v75_native_ternary_spark2" >&2
  exit 1
fi
if [[ "$OUT_LOCAL" == "$REMOTE_OUT" ]]; then
  # same relative path would still be ok on different machines, but refuse if
  # someone points OUT_LOCAL at the live local train tree
  :
fi

mkdir -p "$OUT_LOCAL"
LOG="$OUT_LOCAL/pull_waiter.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

if [[ ! -f "$KEY" ]]; then
  log "ERROR: missing NVIDIA Sync key: $KEY"
  exit 1
fi

log "watching $HOST:~/$REMOTE_DIR/$REMOTE_OUT"
log "  will rsync → $OUT_LOCAL/  (local $LOCAL_NATIVE left alone)"

while true; do
  if out=$("${SSH[@]}" "bash -s" <<EOS 2>/dev/null
set -e
cd "\$HOME/$REMOTE_DIR/$REMOTE_OUT" 2>/dev/null || exit 2
PID=\$(cat train.pid 2>/dev/null || true)
if [[ -n "\$PID" ]] && kill -0 "\$PID" 2>/dev/null; then
  if [[ -f progress.json ]]; then
    python3 -c "import json; d=json.load(open('progress.json')); print(f\\"step={d.get('step')}/{d.get('steps_total')} best={d.get('best_val')} avg={d.get('avg_loss')} tok/s={d.get('tok_s')}\\")"
  else
    echo "alive (no progress.json yet)"
  fi
  exit 0
fi
if [[ -f best.pt ]]; then
  echo "DONE"
  exit 1
fi
echo "DEAD_NO_BEST"
exit 3
EOS
  ); then
    log "  still running: $out"
    sleep 120
    continue
  fi
  case "$out" in
    *DONE*)
      log "remote train finished (best.pt present)"
      break
      ;;
    *DEAD_NO_BEST*)
      log "remote train dead without best.pt; dump log"
      "${SSH[@]}" "tail -50 ~/$REMOTE_DIR/$REMOTE_OUT/train.log" | tee -a "$LOG" || true
      exit 1
      ;;
    *)
      log "  poll error ($out); retry in 60s"
      sleep 60
      ;;
  esac
done

# Safety: refuse to rsync into a directory that looks like a live local train
if [[ -f "$LOCAL_NATIVE/train.pid" ]]; then
  lpid=$(cat "$LOCAL_NATIVE/train.pid" 2>/dev/null || true)
  if [[ -n "$lpid" ]] && kill -0 "$lpid" 2>/dev/null; then
    log "note: local native train still running (pid $lpid) under $LOCAL_NATIVE — not touching it"
  fi
fi
if [[ -f "$OUT_LOCAL/train.pid" ]]; then
  opid=$(cat "$OUT_LOCAL/train.pid" 2>/dev/null || true)
  if [[ -n "$opid" ]] && kill -0 "$opid" 2>/dev/null; then
    log "ERROR: $OUT_LOCAL has a live train.pid ($opid); abort rsync to avoid clobber"
    exit 1
  fi
fi

log "rsync from spark2:$REMOTE_OUT/ → $OUT_LOCAL/"
rsync -avz --progress \
  -e "ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=20" \
  "$HOST:$REMOTE_DIR/$REMOTE_OUT/" \
  "$OUT_LOCAL/"

if [[ ! -f "$OUT_LOCAL/best.pt" ]]; then
  log "ERROR: missing $OUT_LOCAL/best.pt after rsync"
  exit 1
fi

# Marker so we never confuse trees
echo "source=spark2 host=$HOST remote=$REMOTE_OUT pulled=$(date -Iseconds)" \
  >"$OUT_LOCAL/SOURCE_SPARK2.txt"

PY="${PY:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

log "export L3T2 → $OUT_LOCAL/best.ll3m"
"$PY" scripts/export_weights.py \
  --checkpoint "$OUT_LOCAL/best.pt" \
  --out "$OUT_LOCAL/best.ll3m" \
  --quant ternary \
  --tokenizer data/tokenizer.json \
  2>&1 | tee -a "$LOG"

if [[ -f checkpoints/v75/best.pt ]]; then
  log "quality compare vs local fp32 v75"
  "$PY" scripts/compare_fp32_vs_qat.py \
    --baseline checkpoints/v75/best.pt \
    --qat "$OUT_LOCAL/best.pt" \
    --out "$OUT_LOCAL/quality_compare_vs_fp32.json" \
    2>&1 | tee -a "$LOG" || log "compare vs fp32 failed (non-fatal)"
fi
if [[ -f checkpoints/v75_ternary/best.pt ]]; then
  log "quality compare vs local posthoc QAT"
  "$PY" scripts/compare_fp32_vs_qat.py \
    --baseline checkpoints/v75_ternary/best.pt \
    --qat "$OUT_LOCAL/best.pt" \
    --out "$OUT_LOCAL/quality_compare_vs_posthoc.json" \
    2>&1 | tee -a "$LOG" || log "compare vs posthoc failed (non-fatal)"
fi
if [[ -f "$LOCAL_NATIVE/best.pt" ]]; then
  log "quality compare vs local ROCm native ($LOCAL_NATIVE)"
  "$PY" scripts/compare_fp32_vs_qat.py \
    --baseline "$LOCAL_NATIVE/best.pt" \
    --qat "$OUT_LOCAL/best.pt" \
    --out "$OUT_LOCAL/quality_compare_vs_local_native.json" \
    2>&1 | tee -a "$LOG" || log "compare vs local native failed (non-fatal)"
fi

log "done. spark2 weights in $OUT_LOCAL/ (local $LOCAL_NATIVE untouched)"
ls -lah "$OUT_LOCAL/best.pt" "$OUT_LOCAL/best.ll3m" 2>&1 | tee -a "$LOG"
log "serve spark2 pack: python -m ll3m.chat --checkpoint $OUT_LOCAL/best.ll3m chat --mode story"
