#!/usr/bin/env bash
# Tool SFT on top of a ternary (or fp32) checkpoint.
# Research: Kimi agentic data + When2Call balance (gen_tool_sft no-call mix).
#
# Keep QAT on when the base was ternary so serve grid does not drift:
#   CKPT=checkpoints/v75_native_ternary/best.pt OUT=checkpoints/v75_native_tool \
#     bash scripts/train_tool_sft.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"

CKPT="${CKPT:-checkpoints/v75_native_ternary/best.pt}"
OUT="${OUT:-checkpoints/v75_native_tool}"
STEPS="${STEPS:-3000}"
DATA_TXT="${DATA_TXT:-data/tool_sft.txt}"
DATA_BIN="${DATA_BIN:-data/tool_sft.bin}"
TOK="${TOK:-data/tokenizer.json}"
# Match product pack when base is ternary (default on for native path).
# Dense ladder / product path keeps QAT on all the way; refuse accidental drop.
QAT="${QAT:-ternary}"
EMBED_QAT="${EMBED_QAT:-int8}"
N_EP="${N_EP:-6000}"
DEVICE="${DEVICE:-rocm}"
# spark2 / small-L3 hosts need this; product gate is 9950X serve, not remote train
FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]] || [[ "$DEVICE" == "cuda" ]]; then
  FORCE_FLAG=(--force)
fi

if [[ "$QAT" == "none" || "$EMBED_QAT" == "none" ]]; then
  echo "ERROR: refuse tool SFT with QAT off (QAT=$QAT EMBED_QAT=$EMBED_QAT)" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "missing $CKPT (set CKPT=...)" >&2
  exit 1
fi

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: no GPU torch (need ROCm build)" >&2
  exit 1
fi

echo "=== gen tool SFT (When2Call-balanced) → $DATA_TXT  n=$N_EP ==="
"$PY" scripts/gen_tool_sft.py --out "$DATA_TXT" --n "$N_EP"

echo "=== encode BPE → $DATA_BIN ==="
"$PY" - <<PY
from pathlib import Path
import numpy as np
from ll3m.tokenizer import BPETokenizer
tok = BPETokenizer.load("$TOK")
text = Path("$DATA_TXT").read_text(encoding="utf-8")
ids = tok.encode(text)
arr = np.array(ids, dtype=np.uint16)
Path("$DATA_BIN").write_bytes(arr.tobytes())
print(f"encoded {len(ids):,} tokens → $DATA_BIN")
PY

# Absolute step target = resume step + STEPS
RESUME_STEP=$("$PY" -c "import torch; print(int(torch.load('$CKPT', map_location='cpu', weights_only=False).get('step', 0)))")
TOTAL=$((RESUME_STEP + STEPS))
echo "=== tool SFT resume=$CKPT step $RESUME_STEP → abs $TOTAL  qat=$QAT embed=$EMBED_QAT ==="

mkdir -p "$OUT"
"$PY" -u -m ll3m.train \
  --resume "$CKPT" \
  --data "$DATA_TXT" \
  --tokenizer "$TOK" \
  --out-dir "$OUT" \
  --steps "$TOTAL" \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 5e-5 \
  --warmup 100 \
  --dropout 0.05 \
  --weight-decay 0.1 \
  --optimizer lion \
  --eval-every 200 \
  --log-every 50 \
  --patience 0 \
  --device "$DEVICE" \
  --gpu-mem-fraction 0.45 \
  --serve-quant ternary \
  "${FORCE_FLAG[@]}" \
  --qat "$QAT" \
  --embed-qat "$EMBED_QAT" \
  --z-loss 1e-4 \
  --mtp 0.05 \
  --mtp-depth 2 \
  --layer-scale-init 1e-4 \
  --prompt "User: What is 6 times 7?"$'\n'"Assistant:" \
  --sample-tokens 80

echo "=== export L3T2 ==="
"$PY" scripts/export_weights.py \
  --checkpoint "$OUT/best.pt" \
  --out "$OUT/best.ll3m" \
  --quant ternary \
  --tokenizer "$TOK"

echo "done → $OUT"
echo "  chat: python -m ll3m.chat --checkpoint $OUT/best.ll3m chat --mode agent --no-assist-tools"
