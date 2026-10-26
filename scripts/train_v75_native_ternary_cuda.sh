#!/usr/bin/env bash
# Native ternary train for NVIDIA CUDA hosts (spark2 / GB10).
# Always runs inside GNU screen so SSH drop does not kill the job.
#
#   bash scripts/train_v75_native_ternary_cuda.sh
#   screen -r ll3m-native     # attach
#   Ctrl-A d                  # detach
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-checkpoints/v75_native_ternary}"
STEPS="${STEPS:-50000}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-3000}"
BS="${BS:-8}"
ACCUM="${ACCUM:-8}"
MEM="${MEM:-0.90}"
SCREEN_NAME="${SCREEN_NAME:-ll3m-native}"
# If already inside the intended screen session, run train in foreground
INNER="${INNER:-0}"

mkdir -p "$OUT"
LOG="$OUT/train.log"

run_train() {
  echo "[$(date -Iseconds)] native ternary on CUDA → $OUT steps=$STEPS lr=$LR bs=$BS accum=$ACCUM"
  exec "$PY" -u -m ll3m.train \
    --tokenizer data/tokenizer.json \
    --n-layer 23 --n-embd 576 --n-head 12 --n-kv-head 1 \
    --block-size 256 \
    --layer-scale-init 1e-4 \
    --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
    --logit-softcap 0 \
    --batch-size "$BS" --grad-accum "$ACCUM" --gpu-mem-fraction "$MEM" \
    --dropout 0.1 --weight-decay 0.1 --optimizer lion \
    --lr "$LR" --warmup "$WARMUP" --steps "$STEPS" --patience 0 \
    --grad-clip 1.0 --seed 42 \
    --qat ternary --embed-qat int8 \
    --serve-quant ternary \
    --device cuda --infer-scope ccd \
    --force \
    --eval-every 500 --log-every 50 \
    --prompt "Once upon a time" --sample-tokens 120 \
    --out-dir "$OUT"
}

if [[ "$INNER" == "1" ]]; then
  # Log to file and stdout (screen scrollback)
  run_train 2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "ERROR: screen not installed (apt install screen)" >&2
  exit 1
fi

# Avoid duplicate sessions
if screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "screen session '$SCREEN_NAME' already exists:"
  screen -list
  echo "Attach: screen -r $SCREEN_NAME"
  echo "Or: SCREEN_NAME=ll3m-native2 bash $0"
  exit 1
fi

# Write a small runner so screen gets a clean argv
RUNNER="$OUT/screen_run.sh"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INNER=1
export OUT="$OUT" STEPS="$STEPS" LR="$LR" WARMUP="$WARMUP" BS="$BS" ACCUM="$ACCUM" MEM="$MEM" PY="$PY"
exec bash "$ROOT/scripts/train_v75_native_ternary_cuda.sh"
EOF
chmod +x "$RUNNER"

screen -dmS "$SCREEN_NAME" bash "$RUNNER"
sleep 1
# Capture python pid once it starts
for _ in $(seq 1 30); do
  PP=$(pgrep -f "ll3m.train.*$OUT" | head -1 || true)
  if [[ -n "${PP}" ]]; then
    echo "$PP" >"$OUT/train.pid"
    break
  fi
  sleep 0.5
done
echo "started screen session: $SCREEN_NAME"
echo "  attach:  screen -r $SCREEN_NAME"
echo "  log:     $LOG"
echo "  pidfile: $OUT/train.pid  ($(cat "$OUT/train.pid" 2>/dev/null || echo pending))"
screen -list || true
