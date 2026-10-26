#!/usr/bin/env bash
# Two-phase dense ternary product train (legacy path):
#   Phase 1: fp32 base on GPT-4 mix (capability)
#   Phase 2: long ternary QAT (serve grid), same data/objectives
#
# For native ternary from step 0 (BitNet-style product path), use:
#   scripts/train_v75_native_ternary.sh
# Why keep ternary for all of phase 2+: match C pack; tools later stay under QAT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${ROOT}/.venv/bin/python"

OUT1="${OUT1:-checkpoints/v75}"
OUT2="${OUT2:-checkpoints/v75_ternary}"
STEPS1="${STEPS1:-50000}"
STEPS2="${STEPS2:-8000}"
# Product freeze: L23×d576 MQA, ~78.8M params, ternary+int8 WS ~30.32/30.40 MiB
LAYERS="${LAYERS:-23}"
EMBD="${EMBD:-576}"
HEADS="${HEADS:-12}"
KV="${KV:-1}"
BS="${BS:-4}"
ACCUM="${ACCUM:-16}"
MEM="${MEM:-0.45}"
LR1="${LR1:-3e-4}"
LR2="${LR2:-5e-5}"
WARMUP="${WARMUP:-2000}"

common=(
  --arch v2 --tokenizer-mode bpe
  --n-layer "$LAYERS" --n-embd "$EMBD" --n-head "$HEADS" --n-kv-head "$KV"
  --block-size 256
  --layer-scale-init 1e-4
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2
  --batch-size "$BS" --grad-accum "$ACCUM" --gpu-mem-fraction "$MEM"
  --dropout 0.1 --weight-decay 0.1 --optimizer lion
  --device rocm --infer-scope ccd --serve-quant ternary
  --eval-every 500 --log-every 50
  --prompt "Once upon a time" --sample-tokens 120
)

echo "=== Phase 1: fp32 base → $OUT1  steps=$STEPS1 ==="
"$PY" -m ll3m.train \
  "${common[@]}" \
  --qat none \
  --lr "${LR1:-3e-4}" --warmup "${WARMUP:-2000}" --steps "$STEPS1" --patience 0 \
  --out-dir "$OUT1"

# --steps is absolute (resume continues past phase-1 step index)
TOTAL2=$((STEPS1 + STEPS2))
echo "=== Phase 2: ternary Linear + int8 embed QAT → $OUT2  abs steps=$TOTAL2 (+$STEPS2) ==="
"$PY" -m ll3m.train \
  "${common[@]}" \
  --qat ternary --embed-qat int8 \
  --resume "$OUT1/best.pt" \
  --lr "$LR2" --warmup 200 --steps "$TOTAL2" --patience 0 \
  --out-dir "$OUT2"

echo "done. serve candidate: $OUT2/best.pt"
echo "export: $PY scripts/export_weights.py --checkpoint $OUT2/best.pt --tokenizer data/tokenizer.json --out $OUT2/best.clm"
