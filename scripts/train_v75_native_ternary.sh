#!/usr/bin/env bash
# Native ternary product train (BitNet-style) on local ROCm.
# Runs inside GNU screen by default (SCREEN=0 to run foreground).
#
#   bash scripts/train_v75_native_ternary.sh
#   screen -r ll3m-native-rocm
#
# For NVIDIA spark2 use: scripts/train_v75_native_ternary_cuda.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"

OUT="${OUT:-checkpoints/v75_native_ternary}"
STEPS="${STEPS:-50000}"
LAYERS="${LAYERS:-23}"
EMBD="${EMBD:-576}"
HEADS="${HEADS:-12}"
KV="${KV:-1}"
BS="${BS:-4}"
ACCUM="${ACCUM:-16}"
MEM="${MEM:-0.45}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-3000}"
DROPOUT="${DROPOUT:-0.1}"
SOFTCAP="${SOFTCAP:-0}"
ZLOSS="${ZLOSS:-1e-4}"
MTP="${MTP:-0.1}"
MTP_DEPTH="${MTP_DEPTH:-2}"
LS="${LS:-1e-4}"
SEED="${SEED:-42}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SCREEN_NAME="${SCREEN_NAME:-ll3m-native-rocm}"
USE_SCREEN="${USE_SCREEN:-1}"
INNER="${INNER:-0}"

mkdir -p "$OUT"
LOG="$OUT/train.log"

if ! "$PY" -c "import torch; assert torch.cuda.is_available(), torch.__version__" 2>/dev/null; then
  echo "ERROR: no GPU torch. Restore ROCm e.g.:" >&2
  echo "  uv pip install --python $PY 'torch==2.10.0+rocm7.0' --index-url https://download.pytorch.org/whl/rocm7.0" >&2
  exit 1
fi

run_train() {
  cat >"$OUT/recipe.json" <<EOF
{
  "path": "native_ternary_rocm",
  "qat": "ternary",
  "embed_qat": "int8",
  "steps": $STEPS,
  "lr": $LR,
  "warmup": $WARMUP,
  "screen": "$SCREEN_NAME"
}
EOF
  echo "[$(date -Iseconds)] Native ternary ROCm → $OUT steps=$STEPS"
  exec "$PY" -u -m ll3m.train \
    --tokenizer data/tokenizer.json \
    --n-layer "$LAYERS" --n-embd "$EMBD" --n-head "$HEADS" --n-kv-head "$KV" \
    --block-size 256 \
    --layer-scale-init "$LS" \
    --z-loss "$ZLOSS" --mtp "$MTP" --mtp-depth "$MTP_DEPTH" \
    --logit-softcap "$SOFTCAP" \
    --batch-size "$BS" --grad-accum "$ACCUM" --gpu-mem-fraction "$MEM" \
    --dropout "$DROPOUT" --weight-decay 0.1 --optimizer lion \
    --lr "$LR" --warmup "$WARMUP" --steps "$STEPS" --patience 0 \
    --grad-clip "$GRAD_CLIP" \
    --seed "$SEED" \
    --qat ternary --embed-qat int8 \
    --serve-quant ternary \
    --device rocm --infer-scope ccd \
    --eval-every 500 --log-every 50 \
    --prompt "Once upon a time" --sample-tokens 120 \
    --out-dir "$OUT"
}

if [[ "$INNER" == "1" ]]; then
  run_train 2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi

if [[ "$USE_SCREEN" != "1" ]]; then
  run_train 2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "WARNING: screen missing; running with nohup instead" >&2
  nohup bash -c "INNER=1 USE_SCREEN=0 bash '$0'" >>"$LOG" 2>&1 &
  echo $! >"$OUT/train.pid"
  exit 0
fi

if screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "screen '$SCREEN_NAME' already running; attach: screen -r $SCREEN_NAME"
  exit 1
fi

RUNNER="$OUT/screen_run.sh"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export INNER=1
export OUT="$OUT" STEPS="$STEPS" LR="$LR" WARMUP="$WARMUP" BS="$BS" ACCUM="$ACCUM" MEM="$MEM" PY="$PY"
export LAYERS="$LAYERS" EMBD="$EMBD" HEADS="$HEADS" KV="$KV" DROPOUT="$DROPOUT"
export SOFTCAP="$SOFTCAP" ZLOSS="$ZLOSS" MTP="$MTP" MTP_DEPTH="$MTP_DEPTH" LS="$LS" SEED="$SEED" GRAD_CLIP="$GRAD_CLIP"
exec bash "$ROOT/scripts/train_v75_native_ternary.sh"
EOF
chmod +x "$RUNNER"
screen -dmS "$SCREEN_NAME" bash "$RUNNER"
sleep 1
for _ in $(seq 1 30); do
  PP=$(pgrep -f "ll3m.train.*out-dir $OUT|ll3m.train.*$OUT" | head -1 || true)
  # broader match
  PP=$(pgrep -f "python -u -m ll3m.train" | head -1 || true)
  if [[ -n "${PP}" ]]; then
    echo "$PP" >"$OUT/train.pid"
    break
  fi
  sleep 0.5
done
echo "started screen: $SCREEN_NAME  attach: screen -r $SCREEN_NAME"
echo "log: $LOG  pid: $(cat "$OUT/train.pid" 2>/dev/null || echo pending)"
