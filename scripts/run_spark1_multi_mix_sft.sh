#!/usr/bin/env bash
# Multi-mix SFT from v76_dense_c: dense replay + fact Q&A + tools, QAT on.
# Then MC vs Qwen4 + agent battery on C / D / multi-mix.
#
# On spark1:
#   bash scripts/run_spark1_multi_mix_sft.sh
# Or screen:
#   screen -dmS ll3m-mix bash -lc 'cd ~/Desktop/cachelm && bash scripts/run_spark1_multi_mix_sft.sh'
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
PY="${PY:-$ROOT/.venv/bin/python}"
BASE="${BASE:-checkpoints/v76_dense_c/best.pt}"
OUT="${OUT:-checkpoints/v76_dense_mix}"
TOK="${TOK:-data/base_en_dense/tokenizer.json}"
DATA="${DATA:-data/multi_mix_sft.txt}"
VAL="${VAL:-data/multi_mix_sft_val.txt}"
STEPS_DELTA="${STEPS_DELTA:-5000}"
BENCH_LIMIT="${BENCH_LIMIT:-400}"
LOG_DIR=checkpoints/dense_pipeline
mkdir -p "$LOG_DIR" "$OUT" data
LOG="$LOG_DIR/multi_mix.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== multi-mix SFT host=$(hostname) base=$BASE out=$OUT ==="
test -x "$PY"
test -f "$BASE"
test -f "$TOK"
test -f data/base_en_dense/mix_train.txt

if [[ ! -f "$DATA" ]]; then
  log "=== build multi-mix data ==="
  "$PY" -u scripts/build_multi_mix_sft.py \
    --dense data/base_en_dense/mix_train.txt \
    --out "$DATA" \
    --val-out "$VAL" \
    --total-chars "${TOTAL_CHARS:-40000000}" \
    --dense-frac 0.50 --fact-frac 0.20 --tool-frac 0.30 \
    2>&1 | tee -a "$LOG"
fi
test -f "$DATA"

RESUME_STEP=$("$PY" -c "import torch; print(int(torch.load('$BASE',map_location='cpu',weights_only=False).get('step',0)))")
TOTAL=$((RESUME_STEP + STEPS_DELTA))
log "resume step=$RESUME_STEP → $TOTAL  QAT=ternary embed=int8"

if [[ ! -f "$OUT/best.pt" ]]; then
  "$PY" -u -m ll3m.train \
    --resume "$BASE" \
    --data "$DATA" \
    --val-data "$VAL" \
    --tokenizer "$TOK" \
    --out-dir "$OUT" \
    --steps "$TOTAL" \
    --batch-size 4 --grad-accum 8 \
    --lr 3e-5 --warmup 100 --dropout 0.05 \
    --weight-decay 0.1 --optimizer lion \
    --eval-every 200 --log-every 50 --patience 0 \
    --device cuda --gpu-mem-fraction 0.50 --force \
    --serve-quant ternary --qat ternary --embed-qat int8 \
    --z-loss 1e-4 --mtp 0.05 --mtp-depth 2 \
    --layer-scale-init 1e-4 \
    --prompt $'User: What is the capital of France?\nAssistant:' \
    --sample-tokens 60 \
    2>&1 | tee -a "$OUT/train.log" | tee -a "$LOG"
else
  log "reuse $OUT/best.pt"
fi

# ensure best
if [[ ! -f "$OUT/best.pt" && -f "$OUT/last.pt" ]]; then
  cp -f "$OUT/last.pt" "$OUT/best.pt"
  log "promoted last.pt → best.pt"
fi
test -f "$OUT/best.pt"

"$PY" -c "
import torch
d=torch.load('$OUT/best.pt', map_location='cpu', weights_only=False)
assert d.get('qat')=='ternary' and d.get('embed_qat')=='int8', d.get('qat')
print('QAT_OK', d.get('step'), d.get('qat'), d.get('embed_qat'))
" | tee -a "$LOG"

log "=== export L3T2 ==="
"$PY" scripts/export_weights.py \
  --checkpoint "$OUT/best.pt" \
  --out "$OUT/best.ll3m" \
  --quant ternary \
  --tokenizer "$TOK" \
  2>&1 | tee -a "$LOG"

log "=== MC vs Qwen4 (multi-mix) ==="
"$PY" scripts/validate_vs_qwen4.py \
  --checkpoint "$OUT/best.pt" \
  --tokenizer "$TOK" \
  --tag expMix \
  --limit "$BENCH_LIMIT" \
  --device cuda \
  2>&1 | tee -a "$LOG" || log "bench mix failed"

log "=== agent batteries (torch) ==="
for pair in \
  "C:checkpoints/v76_dense_c/best.pt" \
  "D:checkpoints/v76_dense_tool/best.pt" \
  "MIX:$OUT/best.pt"
do
  label="${pair%%:*}"
  ck="${pair#*:}"
  if [[ ! -f "$ck" ]]; then
    log "skip battery $label (missing $ck)"
    continue
  fi
  outj="$(dirname "$ck")/battery_${label}.json"
  log "battery $label → $outj"
  "$PY" scripts/eval_agent_battery.py \
    --checkpoint "$ck" \
    --python \
    --out "$outj" \
    2>&1 | tee -a "$LOG" || log "battery $label failed"
done

log "=== multi-mix pipeline complete ==="
echo "HOST=$(hostname) END=$(date -Iseconds)" | tee "$LOG_DIR/MULTI_MIX_DONE.txt"
# summary table
"$PY" - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
print("=== MC macros ===")
for tag, p in [
    ("C", "checkpoints/v76_dense_c/vs_qwen4_expC.json"),
    ("D", "checkpoints/v76_dense_tool/vs_qwen4_expD.json"),
    ("MIX", "checkpoints/v76_dense_mix/vs_qwen4_expMix.json"),
]:
    path = Path(p)
    if not path.is_file():
        print(f"  {tag}: missing")
        continue
    d = json.loads(path.read_text())
    o, q = d.get("ours_macro"), d.get("qwen4_macro")
    print(f"  {tag:4} ours={o:.2f}  qwen={q:.2f}")
print("=== battery pass rates ===")
for tag in ("C", "D", "MIX"):
    for p in Path("checkpoints").rglob(f"battery_{tag}.json"):
        d = json.loads(p.read_text())
        ok = d.get("passed")
        n = d.get("total")
        rate = d.get("pass_rate")
        print(f"  {tag:4} {p}: {ok}/{n} ({100*(rate or 0):.1f}%)")
PY
