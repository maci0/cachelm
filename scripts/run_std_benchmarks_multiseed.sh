#!/usr/bin/env bash
# Multi-seed MC wrapper (P1). Full 7-task n=400 × S seeds is multi-hour on CPU.
# Example (SciQ-focused / small limit for smoke):
#   LIMIT=50 SEEDS="42 43 44" bash scripts/run_std_benchmarks_multiseed.sh \
#     checkpoints/v76_dense_c/best.pt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CKPT="${1:-checkpoints/v76_dense_c/best.pt}"
LIMIT="${LIMIT:-100}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cpu}"
TOK="${TOK:-data/tokenizer.json}"
TASKS="${TASKS:-hellaswag,arc_easy,winogrande,boolq,openbookqa,sciq,copa}"
OUTDIR="${OUTDIR:-checkpoints/compare/multiseed}"
mkdir -p "$OUTDIR"
for s in $SEEDS; do
  out="$OUTDIR/$(basename "$(dirname "$CKPT")")_seed${s}_limit${LIMIT}.json"
  echo "=== seed=$s limit=$LIMIT -> $out ==="
  .venv/bin/python scripts/run_std_benchmarks.py \
    --checkpoint "$CKPT" \
    --tokenizer "$TOK" \
    --tasks "$TASKS" \
    --limit "$LIMIT" --seed "$s" --device "$DEVICE" \
    --out "$out"
done
echo "done; aggregate macros with jq '.macro_acc_pct' $OUTDIR/*.json"
