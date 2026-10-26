#!/usr/bin/env bash
# End-to-end on spark2 (or any CUDA host): deps → data → native ternary train.
# Intended to run inside screen so the whole pipeline survives disconnect.
#
#   SCREEN_NAME=ll3m-base-en bash scripts/run_v76_base_en_pipeline_cuda.sh
#
# Local reboot does NOT affect this job if it already runs on spark2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Project-local HF cache (spark2 ~/.cache/huggingface/hub is often root-owned)
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-checkpoints/v76_base_en}"
DATA_DIR="${DATA_DIR:-data/base_en}"
RAW="${RAW:-$DATA_DIR/raw}"
FINEWEB_CHARS="${FINEWEB_CHARS:-1500000000}"
SYN_CHARS="${SYN_CHARS:-250000000}"
STORIES_CHARS="${STORIES_CHARS:-0}"
STEPS="${STEPS:-60000}"
SCREEN_NAME="${SCREEN_NAME:-ll3m-base-en}"
INNER="${INNER:-0}"
LOG_DIR="$OUT"
mkdir -p "$OUT" "$RAW" "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"
PIPE_LOG="$OUT/pipeline.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$PIPE_LOG"; }

run_pipeline() {
  log "=== v76 base-English pipeline host=$(hostname) ==="
  log "OUT=$OUT STEPS=$STEPS FINEWEB_CHARS=$FINEWEB_CHARS"

  if [[ ! -x "$PY" ]]; then
    log "ERROR: python not found at $PY"
    exit 1
  fi

  log "install datasets + huggingface_hub if needed"
  if ! "$PY" -c "import datasets" 2>/dev/null; then
    log "datasets missing; installing into venv..."
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "$PY" "datasets>=2.19" "huggingface_hub>=0.23" 2>&1 | tee -a "$PIPE_LOG" || true
    fi
    if ! "$PY" -c "import datasets" 2>/dev/null; then
      "$PY" -m pip install "datasets>=2.19" "huggingface_hub>=0.23" 2>&1 | tee -a "$PIPE_LOG"
    fi
  fi
  if ! "$PY" -c "import datasets" 2>/dev/null; then
    log "ERROR: could not import datasets after install"
    exit 1
  fi
  "$PY" -c "import datasets, torch; print('datasets', datasets.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$PIPE_LOG"

  if [[ ! -f "$DATA_DIR/mix_train.bin" || ! -f "$DATA_DIR/tokenizer.json" ]]; then
    log "=== download raw base-English sources ==="
    "$PY" -u scripts/download_base_english.py \
      --out-dir "$RAW" \
      --fineweb-chars "$FINEWEB_CHARS" \
      --synthetic-chars "$SYN_CHARS" \
      --stories-chars "$STORIES_CHARS" \
      2>&1 | tee -a "$PIPE_LOG"

    log "=== prepare mix + BPE ==="
    "$PY" -u scripts/prepare_base_english.py \
      --raw-dir "$RAW" \
      --out-dir "$DATA_DIR" \
      --vocab-size 2048 \
      2>&1 | tee -a "$PIPE_LOG"
  else
    log "reuse existing $DATA_DIR mix + tokenizer"
  fi

  if [[ ! -f "$DATA_DIR/mix_train.bin" ]]; then
    log "ERROR: data prep failed (no mix_train.bin)"
    exit 1
  fi

  log "=== native ternary train ==="
  # Run train in foreground inside this pipeline (already under screen)
  export INNER=1
  export OUT DATA="$DATA_DIR/mix_train.txt" VAL="$DATA_DIR/mix_val.txt" TOK="$DATA_DIR/tokenizer.json"
  export STEPS LR="${LR:-3e-4}" WARMUP="${WARMUP:-3000}" BS="${BS:-8}" ACCUM="${ACCUM:-8}" MEM="${MEM:-0.90}" PY
  bash "$ROOT/scripts/train_v76_base_en_cuda.sh" 2>&1 | tee -a "$PIPE_LOG"
  log "pipeline finished exit=$?"
}

if [[ "$INNER" == "1" ]]; then
  run_pipeline
  exit 0
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "ERROR: need screen" >&2
  exit 1
fi

if screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "screen '$SCREEN_NAME' already running:"
  screen -list
  exit 1
fi

# Use bash -lc so login env is set and the session stays until pipeline exits.
screen -dmS "$SCREEN_NAME" bash -lc "
  cd '$ROOT' || exit 1
  export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}'
  export INNER=1
  export OUT='$OUT' DATA_DIR='$DATA_DIR' RAW='$RAW'
  export FINEWEB_CHARS='$FINEWEB_CHARS' SYN_CHARS='$SYN_CHARS' STORIES_CHARS='$STORIES_CHARS'
  export STEPS='$STEPS' LR='${LR:-3e-4}' WARMUP='${WARMUP:-3000}' BS='${BS:-8}' ACCUM='${ACCUM:-8}' MEM='${MEM:-0.90}' PY='$PY'
  bash '$ROOT/scripts/run_v76_base_en_pipeline_cuda.sh'
  ec=\$?
  echo PIPE_EXIT=\$ec >>'$PIPE_LOG'
  exit \$ec
"
sleep 2
echo "started remote-safe pipeline screen: $SCREEN_NAME"
echo "  host-local attach: screen -r $SCREEN_NAME"
echo "  log: $PIPE_LOG"
echo "  THIS JOB DOES NOT DEPEND ON YOUR LAPTOP / LOCAL MACHINE"
echo "HOST=$(hostname) START=$(date -Iseconds) SCREEN=$SCREEN_NAME" | tee "$OUT/RUN_ON_SPARK2.txt"
screen -list || true
