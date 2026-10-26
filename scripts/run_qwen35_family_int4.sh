#!/usr/bin/env bash
# Benchmark Qwen3.5 *Base* family under portable weight-only int4 (not NF4/NVFP4).
# Same MC protocol as LL3M (zero-shot continuation log-likelihood).
#
# Default models (dense Base, fairer closed-book than chat-tuned):
#   0.8B, 2B, 4B, 9B  (+ optional 27B if QWEN_INCLUDE_27B=1)
#
#   bash scripts/run_qwen35_family_int4.sh
#   FORCE_REBENCH=1 bash scripts/run_qwen35_family_int4.sh
#   LIMIT=200 bash scripts/run_qwen35_family_int4.sh   # faster smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
# Avoid root-owned ~/.triton/cache (PermissionError on GB10/torch triton path)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/torchinductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
# Triton CudaUtils needs Python.h; on spark1 system headers may be missing.
# Extract libpython3.12-dev into project tree (no sudo) and wrap gcc.
PYDEV_ROOT="${PYDEV_ROOT:-$ROOT/.cache/python_dev}"
if [[ ! -f "$PYDEV_ROOT/usr/include/python3.12/Python.h" ]]; then
  mkdir -p "$ROOT/.cache/debs" "$PYDEV_ROOT"
  (
    cd "$ROOT/.cache/debs"
    apt-get download libpython3.12-dev >/dev/null 2>&1 || true
    if ls libpython3.12-dev*.deb 1>/dev/null 2>&1; then
      dpkg-deb -x libpython3.12-dev*.deb "$PYDEV_ROOT"
    fi
  )
fi
mkdir -p "$ROOT/.cache/bin"
cat >"$ROOT/.cache/bin/gcc" <<WRAP
#!/bin/bash
INC="$PYDEV_ROOT/usr/include"
exec /usr/bin/gcc \\
  -I"\$INC/python3.12" \\
  -I"\$INC/aarch64-linux-gnu/python3.12" \\
  -I"\$INC" \\
  "\$@"
WRAP
chmod +x "$ROOT/.cache/bin/gcc"
# Prefer wrapper for triton's hardcoded gcc invocations
export PATH="$ROOT/.cache/bin:$PATH"
PY="${PY:-$ROOT/.venv/bin/python}"
LIMIT="${LIMIT:-400}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-checkpoints/compare}"
LOG_DIR=checkpoints/dense_pipeline
mkdir -p "$OUT_DIR" "$LOG_DIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
LOG="$LOG_DIR/qwen35_family_int4.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Prefer Base checkpoints for closed-book MC (matches existing 0.8B baseline).
MODELS=(
  "Qwen/Qwen3.5-0.8B-Base"
  "Qwen/Qwen3.5-2B-Base"
  "Qwen/Qwen3.5-4B-Base"
  "Qwen/Qwen3.5-9B-Base"
)
if [[ "${QWEN_INCLUDE_27B:-0}" == "1" ]]; then
  MODELS+=("Qwen/Qwen3.5-27B")
fi
# Allow override: MODELS_CSV="Qwen/A,Qwen/B"
if [[ -n "${MODELS_CSV:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
fi

log "=== Qwen3.5 family weight-only int4 MC limit=$LIMIT seed=$SEED FORCE_REBENCH=${FORCE_REBENCH:-0} ==="
log "models: ${MODELS[*]}"
log "quant: per-row absmax int4 (NOT bitsandbytes NF4 / NOT NVFP4)"

if ! "$PY" -c "import transformers, datasets, torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "installing transformers + datasets …"
  if command -v uv >/dev/null; then
    uv pip install --python "$PY" transformers accelerate datasets sentencepiece protobuf 2>&1 | tee -a "$LOG"
  else
    "$PY" -m pip install -U transformers accelerate datasets sentencepiece protobuf 2>&1 | tee -a "$LOG"
  fi
fi
"$PY" -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available())" | tee -a "$LOG"

for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  out="$OUT_DIR/${safe}_int4_limit${LIMIT}.json"
  if [[ -f "$out" && "${FORCE_REBENCH:-0}" != "1" ]]; then
    if "$PY" -c "import json; d=json.load(open('$out')); assert d.get('macro_acc_pct'); assert len(d.get('tasks',{}))>=5; assert (d.get('quant') or {}).get('mode')=='weight_only_int4_absmax'" 2>/dev/null; then
      log "skip existing int4 $out"
      continue
    fi
    log "existing $out is not weight-only int4 (or incomplete) → rebench"
  fi
  # drop non-int4 leftovers (old NF4 path names, etc.)
  if [[ -f "$out" ]]; then
    mode="$("$PY" -c "import json; print((json.load(open('$out')).get('quant') or {}).get('mode',''))" 2>/dev/null || true)"
    if [[ "$mode" != "weight_only_int4_absmax" ]]; then
      rm -f "$out"
      log "removed non-int4 $out (was mode=$mode)"
    fi
  fi
  log "=== bench $model → $out (weight-only int4) ==="
  "$PY" -u scripts/run_std_benchmarks_hf.py \
    --model "$model" \
    --weight-int4 \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --device cuda \
    --out "$out" \
    2>&1 | tee -a "$LOG"
done

log "=== aggregate vs LL3M ==="
"$PY" -u scripts/aggregate_qwen_family_vs_ll3m.py \
  --compare-dir "$OUT_DIR" \
  --limit "$LIMIT" \
  --out "$OUT_DIR/ll3m_vs_qwen35_family_int4.json" \
  2>&1 | tee -a "$LOG"

log "DONE family weight-only int4 bench"
echo "HOST=$(hostname) END=$(date -Iseconds) QUANT=weight_only_int4_absmax" | tee "$LOG_DIR/QWEN_FAMILY_DONE.txt"
