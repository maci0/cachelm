#!/usr/bin/env bash
# Native ternary base-English train on CUDA (spark2).
# Always runs inside GNU screen so SSH drop / local reboot does not kill it.
#
#   bash scripts/train_v76_base_en_cuda.sh
#   screen -r ll3m-base-en
#
# Data: data/base_en/{mix_train.txt,mix_val.txt,tokenizer.json}
# Out:  checkpoints/v76_base_en/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-checkpoints/v76_base_en}"
DATA="${DATA:-data/base_en/mix_train.txt}"
VAL="${VAL:-data/base_en/mix_val.txt}"
TOK="${TOK:-data/base_en/tokenizer.json}"
STEPS="${STEPS:-60000}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-3000}"
BS="${BS:-8}"
ACCUM="${ACCUM:-8}"
MEM="${MEM:-0.90}"
SCREEN_NAME="${SCREEN_NAME:-ll3m-base-en}"
INNER="${INNER:-0}"

mkdir -p "$OUT"
LOG="$OUT/train.log"

for f in "$DATA" "$VAL" "$TOK"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f (run base-English data pipeline first)" >&2
    exit 1
  fi
done

run_train() {
  echo "[$(date -Iseconds)] v76 base-English native ternary CUDA → $OUT steps=$STEPS"
  echo "  data=$DATA tok=$TOK"
  exec "$PY" -u -m ll3m.train \
    --data "$DATA" \
    --val-data "$VAL" \
    --tokenizer "$TOK" \
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
    --prompt $'User: Explain gravity in simple terms.\nAssistant:' \
    --sample-tokens 120 \
    --out-dir "$OUT"
}

if [[ "$INNER" == "1" ]]; then
  run_train 2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "ERROR: screen not installed" >&2
  exit 1
fi

if screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "screen session '$SCREEN_NAME' already exists:"
  screen -list
  echo "Attach: screen -r $SCREEN_NAME"
  exit 1
fi

RUNNER="$OUT/screen_run.sh"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INNER=1
export OUT="$OUT" DATA="$DATA" VAL="$VAL" TOK="$TOK"
export STEPS="$STEPS" LR="$LR" WARMUP="$WARMUP" BS="$BS" ACCUM="$ACCUM" MEM="$MEM" PY="$PY"
exec bash "$ROOT/scripts/train_v76_base_en_cuda.sh"
EOF
chmod +x "$RUNNER"

screen -dmS "$SCREEN_NAME" bash "$RUNNER"
sleep 1
for _ in $(seq 1 40); do
  PP=$(pgrep -f "ll3m.train.*$OUT" | head -1 || true)
  if [[ -n "${PP}" ]]; then
    echo "$PP" >"$OUT/train.pid"
    break
  fi
  sleep 0.5
done
echo "started screen: $SCREEN_NAME  (survives SSH drop; independent of local machine)"
echo "  attach: screen -r $SCREEN_NAME"
echo "  log:    $LOG"
echo "  pid:    $(cat "$OUT/train.pid" 2>/dev/null || echo pending)"
echo "HOST=$(hostname) START=$(date -Iseconds)" | tee "$OUT/RUN_ON_SPARK2.txt"
screen -list || true
