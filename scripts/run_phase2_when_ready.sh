#!/usr/bin/env bash
# Wait for phase-1 dense train PID or completion, then ternary QAT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT1="${OUT1:-checkpoints/v2_dense}"
OUT2="${OUT2:-checkpoints/v2_dense_ternary}"
PID1="${PID1:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="$ROOT/.venv/bin/python"

if [[ -n "$PID1" ]]; then
  echo "waiting for phase1 pid $PID1 ..."
  while kill -0 "$PID1" 2>/dev/null; do sleep 30; done
  echo "phase1 process exited"
fi

# require a best.pt
if [[ ! -f "$OUT1/best.pt" ]]; then
  echo "missing $OUT1/best.pt" >&2
  exit 1
fi

# read last step from progress if present
STEPS1=25000
if [[ -f "$OUT1/progress.json" ]]; then
  STEPS1=$("$PY" -c "import json; print(json.load(open('$OUT1/progress.json')).get('steps_total',25000))")
fi
STEPS2="${STEPS2:-12000}"
TOTAL2=$((STEPS1 + STEPS2))
mkdir -p "$OUT2"
echo "=== Phase 2 ternary QAT resume $OUT1/best.pt → $OUT2 steps=$TOTAL2 ==="
exec "$PY" -m ll3m.train \
  --arch v2 --tokenizer-mode bpe \
  --n-layer 18 --n-embd 320 --n-head 8 --n-kv-head 2 --block-size 256 \
  --peri-ln --layer-scale-init 1e-2 \
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
  --batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.40 \
  --dropout 0.12 --weight-decay 0.1 --optimizer lion \
  --device rocm --infer-scope ccd --serve-quant ternary \
  --qat ternary --resume "$OUT1/best.pt" \
  --lr 8e-5 --warmup 200 --steps "$TOTAL2" --patience 10 \
  --eval-every 500 --log-every 50 \
  --prompt "Once upon a time" --sample-tokens 120 \
  --out-dir "$OUT2"
