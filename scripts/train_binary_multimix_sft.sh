#!/usr/bin/env bash
# Post-base multi-mix SFT for v77 binary L30 (fair stage after mix_train base).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"
CKPT="${CKPT:-checkpoints/v77_binary_l30/last.pt}"
OUT="${OUT:-checkpoints/v77_binary_l30_sft}"
DATA="${DATA:-data/cap_mix_v15_agent.txt}"
TOK="${TOK:-data/tokenizer.json}"
EXTRA="${EXTRA_STEPS:-10000}"

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "need ROCm torch" >&2; exit 1
fi
STEP=$("$PY" -c "import torch; print(torch.load('$CKPT',map_location='cpu',weights_only=False)['step'])")
END=$((STEP + EXTRA))
mkdir -p "$OUT"
echo "binary multi-mix SFT: resume $CKPT step $STEP → $END" | tee -a "$OUT/train.log"
exec "$PY" -u -m ll3m.train \
  --resume "$CKPT" \
  --data "$DATA" --tokenizer "$TOK" \
  --out-dir "$OUT" \
  --steps "$END" \
  --batch-size 4 --grad-accum 8 \
  --lr 5e-6 --warmup 200 \
  --dropout 0.1 --weight-decay 0.1 \
  --device rocm \
  --qat binary --embed-qat int8 --serve-quant binary \
  --infer-scope ccd --patience 0 \
  --prompt "Once upon a time"
