#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"
CKPT="${CKPT:-checkpoints/v77_binary_l30_sft/last.pt}"
OUT="${OUT:-checkpoints/v77_binary_l30_sciq}"
DATA="${DATA:-data/cap_mix_v16_binary_sciq.txt}"
EXTRA="${EXTRA_STEPS:-12000}"
STEP=$("$PY" -c "import torch; print(torch.load('$CKPT',map_location='cpu',weights_only=False)['step'])")
END=$((STEP + EXTRA))
mkdir -p "$OUT"
echo "binary sciq stage: $CKPT $STEP → $END" | tee -a "$OUT/train.log"
exec "$PY" -u -m ll3m.train \
  --resume "$CKPT" \
  --data "$DATA" --tokenizer data/tokenizer.json \
  --out-dir "$OUT" --steps "$END" \
  --batch-size 4 --grad-accum 8 \
  --lr 5e-6 --warmup 200 \
  --dropout 0.1 --weight-decay 0.1 \
  --device rocm \
  --qat binary --embed-qat int8 --serve-quant binary \
  --infer-scope ccd --patience 0 \
  --prompt "Once upon a time"
