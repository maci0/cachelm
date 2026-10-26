#!/usr/bin/env bash
# Wait for v75 phase-1 (fp32) to finish, then ternary Linear + int8 embed QAT,
# export L3T2 packs, and write a quality comparison report.
#
#   nohup bash scripts/run_v75_ternary_qat_when_ready.sh >> checkpoints/v75_ternary/waiter.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT1="${OUT1:-checkpoints/v75}"
OUT2="${OUT2:-checkpoints/v75_ternary}"
PID1="${PID1:-}"
# Absolute step target = phase1_steps + QAT_STEPS (resume continues from ckpt step)
QAT_STEPS="${QAT_STEPS:-6000}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"
mkdir -p "$OUT2"
LOG="$OUT2/phase2.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT2/waiter.log"; }

# Resolve phase1 PID if not given (match v75 train out-dir)
if [[ -z "$PID1" ]]; then
  PID1=$(pgrep -f "out-dir checkpoints/v75" 2>/dev/null | head -1 || true)
fi

if [[ -n "$PID1" ]] && kill -0 "$PID1" 2>/dev/null; then
  log "waiting for phase1 pid $PID1 ..."
  while kill -0 "$PID1" 2>/dev/null; do
    if [[ -f "$OUT1/progress.json" ]]; then
      step=$("$PY" -c "import json; print(json.load(open('$OUT1/progress.json')).get('step','?'))" 2>/dev/null || echo "?")
      log "  still running  step=$step"
    fi
    sleep 60
  done
  log "phase1 process exited"
else
  log "no live phase1 process (or already done); proceeding if best.pt exists"
fi

# Wait until best.pt is not being written mid-eval (brief settle)
sleep 5

BEST="$OUT1/best.pt"
if [[ ! -f "$BEST" ]]; then
  log "missing $BEST" >&2
  exit 1
fi

# Refuse to start QAT without a working ROCm/HIP torch (common footgun: uv sync
# pulls a CUDA wheel and torch.cuda.is_available() becomes False on AMD).
if ! "$PY" -c "import torch; assert torch.cuda.is_available(), torch.__version__" 2>/dev/null; then
  log "ERROR: $PY has no usable GPU torch (need ROCm build, e.g. torch==2.10.0+rocm7.0)." >&2
  log "  fix: uv pip install --python $PY 'torch==2.10.0+rocm7.0' --index-url https://download.pytorch.org/whl/rocm7.0" >&2
  "$PY" -c "import torch; print('  version=', torch.__version__, 'hip=', getattr(torch.version,'hip',None), 'cuda=', torch.cuda.is_available())" 2>&1 | tee -a "$OUT2/waiter.log" || true
  exit 1
fi
TORCH_INFO=$("$PY" -c "import torch; print(torch.__version__, 'hip='+str(getattr(torch.version,'hip',None)))")
log "GPU torch ok: $TORCH_INFO"

P1_STEPS=$("$PY" -c "import torch; print(int(torch.load('$BEST', map_location='cpu', weights_only=False).get('step', 50000)))")
TOTAL=$((P1_STEPS + QAT_STEPS))
log "=== Phase 2 ternary QAT ==="
log "  resume: $BEST (step $P1_STEPS)"
log "  out:    $OUT2"
log "  steps:  $TOTAL  (+$QAT_STEPS QAT)  qat=ternary embed_qat=int8"

# Same data/mix as phase1 via defaults (mix_train + tokenizer)
nohup "$PY" -u -m ll3m.train \
  --resume "$BEST" \
  --qat ternary \
  --embed-qat int8 \
  --serve-quant ternary \
  --device rocm \
  --gpu-mem-fraction 0.45 \
  --batch-size 4 \
  --grad-accum 16 \
  --dropout 0.1 \
  --lr 5e-5 \
  --warmup 200 \
  --weight-decay 0.1 \
  --optimizer lion \
  --steps "$TOTAL" \
  --patience 0 \
  --eval-every 250 \
  --log-every 50 \
  --infer-scope ccd \
  --prompt "Once upon a time" \
  --sample-tokens 120 \
  --out-dir "$OUT2" \
  >>"$LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" >"$OUT2/train.pid"
log "started QAT train pid=$TRAIN_PID  log=$LOG"

wait "$TRAIN_PID"
RC=$?
log "QAT train exited rc=$RC"
if [[ $RC -ne 0 ]]; then
  log "QAT failed; skip export/compare"
  exit "$RC"
fi

log "export L3T2 packs..."
"$PY" scripts/export_weights.py \
  --checkpoint "$OUT1/best.pt" --out "$OUT1/best.ll3m" --quant ternary \
  >>"$OUT2/waiter.log" 2>&1 || true
"$PY" scripts/export_weights.py \
  --checkpoint "$OUT2/best.pt" --out "$OUT2/best.ll3m" --quant ternary \
  >>"$OUT2/waiter.log" 2>&1

log "quality compare (fp32 masters + posthoc serve vs QAT)..."
"$PY" scripts/compare_fp32_vs_qat.py \
  --baseline "$OUT1/best.pt" \
  --qat "$OUT2/best.pt" \
  --out "$OUT2/quality_compare.json" \
  >>"$OUT2/waiter.log" 2>&1

log "done. report: $OUT2/quality_compare.json"
cat "$OUT2/quality_compare.json" 2>/dev/null || true
