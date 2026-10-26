#!/usr/bin/env bash
# Sequential: ternary-embed QAT then int8-embed QAT from same Linear-ternary base.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="$ROOT/.venv/bin/python"
BASE="${BASE:-checkpoints/v2_dense_ternary/best.pt}"
STEP=$("$PY" -c "import torch; print(int(torch.load('$BASE',map_location='cpu',weights_only=False)['step']))")
EXTRA="${EXTRA_STEPS:-4000}"
TOTAL=$((STEP + EXTRA))

common=(
  --arch v2 --tokenizer-mode bpe
  --n-layer 18 --n-embd 320 --n-head 8 --n-kv-head 2 --block-size 256
  --peri-ln --layer-scale-init 1e-2
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2
  --batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.40
  --dropout 0.12 --weight-decay 0.1 --optimizer lion
  --device rocm --infer-scope ccd --serve-quant ternary
  --qat ternary
  --resume "$BASE"
  --lr 5e-5 --warmup 100 --steps "$TOTAL" --patience 0
  --eval-every 500 --log-every 50
  --prompt "Once upon a time" --sample-tokens 80
)

echo "=== embed ternary QAT  steps=$TOTAL resume=$BASE ==="
mkdir -p checkpoints/exp_embed_ternary
"$PY" -m ll3m.train "${common[@]}" --embed-qat ternary \
  --out-dir checkpoints/exp_embed_ternary

echo "=== embed int8 QAT  steps=$TOTAL resume=$BASE ==="
mkdir -p checkpoints/exp_embed_int8
"$PY" -m ll3m.train "${common[@]}" --embed-qat int8 \
  --out-dir checkpoints/exp_embed_int8

echo "=== compare done ==="
"$PY" scripts/compare_embed_qat.py
