#!/usr/bin/env bash
# Knowledge-density ladder on spark1 (192.168.0.211).
# Survives SSH drop via screen. Local reboot safe.
#
# Stages (env SKIP_*=1 to skip):
#   A already applied in code (kb_facts.json) — validate only
#   B denser synthetic + FineWeb mix + BPE
#   C continue native ternary from v76 best.pt  (QAT ternary + embed int8)
#   D short Q&A SFT + tool SFT                 (QAT stays on)
#   After C and D: std benchmarks + vs Qwen4 JSON
#
# Product pack is locked: Linear ternary STE + embed int8 STE end-to-end.
# Never drop QAT mid-ladder (serve grid drift / L3T2 mismatch).
#
#   INNER=1 bash scripts/run_spark1_dense_pipeline.sh
#   SCREEN_NAME=ll3m-dense bash scripts/run_spark1_dense_pipeline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT_C="${OUT_C:-checkpoints/v76_dense_c}"
OUT_D_QA="${OUT_D_QA:-checkpoints/v76_dense_qa}"
OUT_D_TOOL="${OUT_D_TOOL:-checkpoints/v76_dense_tool}"
DATA_DIR="${DATA_DIR:-data/base_en_dense}"
RAW="${RAW:-$DATA_DIR/raw}"
# Denser mix: more atomic synthetic, less long FineWeb prose (same ~1.2B char budget).
FINEWEB_CHARS="${FINEWEB_CHARS:-500000000}"
SYN_CHARS="${SYN_CHARS:-700000000}"
CONT_STEPS="${CONT_STEPS:-30000}"
QA_STEPS="${QA_STEPS:-3000}"
TOOL_STEPS="${TOOL_STEPS:-3000}"
BENCH_LIMIT="${BENCH_LIMIT:-400}"
BASE_CKPT="${BASE_CKPT:-checkpoints/v76_base_en/best.pt}"
# Locked product QAT (override only if you know why; none is refused)
QAT="${QAT:-ternary}"
EMBED_QAT="${EMBED_QAT:-int8}"
SERVE_QUANT="${SERVE_QUANT:-ternary}"
# prepare density knobs
PREP_MAX_CHARS="${PREP_MAX_CHARS:-1200}"
NEAR_DEDUP_J="${NEAR_DEDUP_J:-0.85}"
SCREEN_NAME="${SCREEN_NAME:-ll3m-dense}"
INNER="${INNER:-0}"
LOG_DIR=checkpoints/dense_pipeline
mkdir -p "$LOG_DIR" "$OUT_C" "$OUT_D_QA" "$OUT_D_TOOL" "$RAW" "$HF_HOME"
LOG="$LOG_DIR/pipeline.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

ensure_best_pt() {
  # $1=out_dir  — promote last.pt → best.pt if best missing (resume-bar miss)
  local dir="$1"
  if [[ -f "$dir/best.pt" ]]; then
    return 0
  fi
  if [[ -f "$dir/last.pt" ]]; then
    cp -f "$dir/last.pt" "$dir/best.pt"
    log "promoted $dir/last.pt → best.pt (no best.pt from train)"
    return 0
  fi
  log "ERROR: neither best.pt nor last.pt in $dir"
  return 1
}

assert_qat_ckpt() {
  # $1=path $2=label  — refuse if checkpoint dropped QAT
  local ck="$1" label="$2"
  "$PY" - "$ck" "$label" "$QAT" "$EMBED_QAT" <<'PY' | tee -a "$LOG"
import sys, torch
from pathlib import Path
ck, label, want_q, want_e = sys.argv[1:5]
p = Path(ck)
if not p.is_file():
    print(f"QAT_ASSERT_FAIL {label}: missing {ck}", flush=True)
    sys.exit(1)
d = torch.load(p, map_location="cpu", weights_only=False)
got_q = str(d.get("qat", "none") or "none")
got_e = str(d.get("embed_qat", "none") or "none")
print(f"QAT_ASSERT {label}: qat={got_q} embed_qat={got_e} step={d.get('step')}", flush=True)
if got_q != want_q or got_e != want_e:
    print(
        f"QAT_ASSERT_FAIL {label}: expected qat={want_q} embed_qat={want_e}, "
        f"got qat={got_q} embed_qat={got_e}",
        flush=True,
    )
    sys.exit(1)
print(f"QAT_ASSERT_OK {label}", flush=True)
PY
}

run() {
  log "=== dense pipeline host=$(hostname) ==="
  log "BASE_CKPT=$BASE_CKPT SYN=$SYN_CHARS FW=$FINEWEB_CHARS CONT_STEPS=$CONT_STEPS"
  log "QAT_LOCK qat=$QAT embed_qat=$EMBED_QAT serve_quant=$SERVE_QUANT (all stages)"

  if [[ "$QAT" == "none" || "$EMBED_QAT" == "none" || "$SERVE_QUANT" == "none" ]]; then
    log "ERROR: QAT must stay on (got QAT=$QAT EMBED_QAT=$EMBED_QAT SERVE_QUANT=$SERVE_QUANT)"
    exit 1
  fi
  if [[ "$QAT" != "ternary" || "$EMBED_QAT" != "int8" ]]; then
    log "WARN: non-product QAT grid QAT=$QAT EMBED_QAT=$EMBED_QAT (product is ternary+int8)"
  fi

  if [[ ! -x "$PY" ]]; then
    log "ERROR: missing $PY — bootstrap venv first"
    exit 1
  fi
  if ! "$PY" -c "import torch,datasets" 2>/dev/null; then
    log "install torch/datasets deps..."
    # assume CUDA venv already prepared on spark1
    "$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  fi

  # ----- Exp A validate (KB load) -----
  if [[ "${SKIP_A:-0}" != "1" ]]; then
    log "=== Exp A: KB size check ==="
    "$PY" - <<'PY' | tee -a "$LOG"
from ll3m.tools import _KB
print(f"KB_KEYS={len(_KB)}")
assert len(_KB) >= 300, len(_KB)
print("ExpA_OK")
PY
  fi
  # Base must already be ternary QAT (never start dense ladder from fp32-only)
  if [[ -f "$BASE_CKPT" ]]; then
    assert_qat_ckpt "$BASE_CKPT" "base"
  fi

  # ----- Exp B: dense data -----
  if [[ "${SKIP_B:-0}" != "1" ]]; then
    if [[ ! -f "$DATA_DIR/mix_train.bin" ]]; then
      log "=== Exp B: download dense sources ==="
      "$PY" -u scripts/download_base_english.py \
        --out-dir "$RAW" \
        --fineweb-chars "$FINEWEB_CHARS" \
        --synthetic-chars "$SYN_CHARS" \
        --stories-chars 0 \
        2>&1 | tee -a "$LOG"
      log "=== Exp B: prepare mix + BPE (max_chars=$PREP_MAX_CHARS near_dedup=$NEAR_DEDUP_J) ==="
      "$PY" -u scripts/prepare_base_english.py \
        --raw-dir "$RAW" \
        --out-dir "$DATA_DIR" \
        --vocab-size 2048 \
        --max-chars "$PREP_MAX_CHARS" \
        --near-dedup-jaccard "$NEAR_DEDUP_J" \
        --prefer-short-weight 1.6 \
        2>&1 | tee -a "$LOG"
    else
      log "reuse existing $DATA_DIR"
    fi
    test -f "$DATA_DIR/mix_train.bin"
    test -f "$DATA_DIR/tokenizer.json"
    log "ExpB_OK tokens=$("$PY" -c "import json;print(json.load(open('$DATA_DIR/manifest.json'))['train_tokens'])")"
  fi

  # ----- Exp C: continue pretrain -----
  if [[ "${SKIP_C:-0}" != "1" ]]; then
    if [[ ! -f "$BASE_CKPT" ]]; then
      log "ERROR: missing BASE_CKPT=$BASE_CKPT"
      exit 1
    fi
    if [[ ! -f "$OUT_C/best.pt" ]]; then
      log "=== Exp C: continue ternary train (QAT=$QAT embed=$EMBED_QAT) ==="
      RESUME_STEP=$("$PY" -c "import torch; print(int(torch.load('$BASE_CKPT',map_location='cpu',weights_only=False).get('step',0)))")
      TOTAL=$((RESUME_STEP + CONT_STEPS))
      log "resume step=$RESUME_STEP → $TOTAL"
      "$PY" -u -m ll3m.train \
        --resume "$BASE_CKPT" \
        --data "$DATA_DIR/mix_train.txt" \
        --val-data "$DATA_DIR/mix_val.txt" \
        --tokenizer "$DATA_DIR/tokenizer.json" \
        --n-layer 23 --n-embd 576 --n-head 12 --n-kv-head 1 \
        --block-size 256 \
        --layer-scale-init 1e-4 \
        --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
        --logit-softcap 0 \
        --batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.90 \
        --dropout 0.1 --weight-decay 0.1 --optimizer lion \
        --lr 1e-4 --warmup 500 --steps "$TOTAL" --patience 0 \
        --grad-clip 1.0 --seed 42 \
        --qat "$QAT" --embed-qat "$EMBED_QAT" \
        --serve-quant "$SERVE_QUANT" \
        --device cuda --infer-scope ccd --force \
        --eval-every 500 --log-every 50 \
        --prompt $'User: What is the capital of France?\nAssistant:' \
        --sample-tokens 80 \
        --out-dir "$OUT_C" \
        2>&1 | tee -a "$OUT_C/train.log" | tee -a "$LOG"
    else
      log "reuse $OUT_C/best.pt"
    fi
    ensure_best_pt "$OUT_C"
    assert_qat_ckpt "$OUT_C/best.pt" "expC"
    "$PY" scripts/export_weights.py \
      --checkpoint "$OUT_C/best.pt" \
      --out "$OUT_C/best.ll3m" \
      --quant ternary \
      --tokenizer "$DATA_DIR/tokenizer.json" \
      2>&1 | tee -a "$LOG"
    log "=== Exp C validate vs Qwen4 ==="
    "$PY" scripts/validate_vs_qwen4.py \
      --checkpoint "$OUT_C/best.pt" \
      --tokenizer "$DATA_DIR/tokenizer.json" \
      --tag expC \
      --limit "$BENCH_LIMIT" \
      --device cuda \
      2>&1 | tee -a "$LOG" || log "bench C failed (non-fatal)"
    log "ExpC_OK"
  fi

  # ----- Exp D: QA SFT then tool SFT -----
  if [[ "${SKIP_D:-0}" != "1" ]]; then
    CKPT_C="$OUT_C/best.pt"
    [[ -f "$CKPT_C" ]] || CKPT_C="$BASE_CKPT"
    # D1 closed-book Q&A from synthetic/KB text only
    if [[ ! -f "$OUT_D_QA/best.pt" ]]; then
      log "=== Exp D1: closed-book Q&A SFT ==="
      "$PY" -u scripts/gen_fact_qa_sft.py --out data/fact_qa_sft.txt --n 12000 2>&1 | tee -a "$LOG"
      # encode bin
      "$PY" - <<PY
from pathlib import Path
import numpy as np
from ll3m.tokenizer import BPETokenizer
tok = BPETokenizer.load("$DATA_DIR/tokenizer.json")
text = Path("data/fact_qa_sft.txt").read_text(encoding="utf-8")
ids = tok.encode(text)
Path("data/fact_qa_sft.bin").write_bytes(np.array(ids, dtype=np.uint16).tobytes())
print(f"encoded {len(ids):,} tokens")
PY
      RESUME_STEP=$("$PY" -c "import torch; print(int(torch.load('$CKPT_C',map_location='cpu',weights_only=False).get('step',0)))")
      TOTAL=$((RESUME_STEP + QA_STEPS))
      log "QA resume=$RESUME_STEP → $TOTAL  QAT=$QAT embed=$EMBED_QAT"
      "$PY" -u -m ll3m.train \
        --resume "$CKPT_C" \
        --data data/fact_qa_sft.txt \
        --tokenizer "$DATA_DIR/tokenizer.json" \
        --out-dir "$OUT_D_QA" \
        --steps "$TOTAL" \
        --batch-size 4 --grad-accum 8 \
        --lr 5e-5 --warmup 100 --dropout 0.05 \
        --weight-decay 0.1 --optimizer lion \
        --eval-every 200 --log-every 50 --patience 0 \
        --device cuda --gpu-mem-fraction 0.45 --force \
        --serve-quant "$SERVE_QUANT" --qat "$QAT" --embed-qat "$EMBED_QAT" \
        --z-loss 1e-4 --mtp 0.05 --mtp-depth 2 \
        --layer-scale-init 1e-4 \
        --prompt $'User: What is the capital of France?\nAssistant:' \
        --sample-tokens 40 \
        2>&1 | tee -a "$OUT_D_QA/train.log" | tee -a "$LOG"
    fi
    ensure_best_pt "$OUT_D_QA"
    assert_qat_ckpt "$OUT_D_QA/best.pt" "expD1_qa"
    # D2 tool SFT
    if [[ ! -f "$OUT_D_TOOL/best.pt" ]]; then
      log "=== Exp D2: tool SFT (QAT=$QAT embed=$EMBED_QAT) ==="
      CKPT_QA="$OUT_D_QA/best.pt"
      [[ -f "$CKPT_QA" ]] || CKPT_QA="$CKPT_C"
      FORCE=1 DEVICE=cuda \
        CKPT="$CKPT_QA" OUT="$OUT_D_TOOL" \
        TOK="$DATA_DIR/tokenizer.json" \
        STEPS="$TOOL_STEPS" N_EP=6000 \
        QAT="$QAT" EMBED_QAT="$EMBED_QAT" \
        bash scripts/train_tool_sft.sh 2>&1 | tee -a "$LOG"
    fi
    ensure_best_pt "$OUT_D_TOOL"
    assert_qat_ckpt "$OUT_D_TOOL/best.pt" "expD2_tool"
    "$PY" scripts/export_weights.py \
      --checkpoint "$OUT_D_TOOL/best.pt" \
      --out "$OUT_D_TOOL/best.ll3m" \
      --quant ternary \
      --tokenizer "$DATA_DIR/tokenizer.json" \
      2>&1 | tee -a "$LOG" || true
    log "=== Exp D validate vs Qwen4 ==="
    "$PY" scripts/validate_vs_qwen4.py \
      --checkpoint "$OUT_D_TOOL/best.pt" \
      --tokenizer "$DATA_DIR/tokenizer.json" \
      --tag expD \
      --limit "$BENCH_LIMIT" \
      --device cuda \
      2>&1 | tee -a "$LOG" || log "bench D failed"
    log "ExpD_OK"
  fi

  log "=== pipeline complete ==="
  echo "HOST=$(hostname) END=$(date -Iseconds)" | tee "$LOG_DIR/DONE.txt"
}

if [[ "$INNER" == "1" ]]; then
  run
  exit 0
fi

if ! command -v screen >/dev/null; then
  echo "need screen" >&2
  exit 1
fi
if screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "screen $SCREEN_NAME already running"
  screen -list
  exit 0
fi
screen -dmS "$SCREEN_NAME" bash -lc "
  cd '$ROOT'
  export INNER=1 CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}'
  export HF_HOME='$HF_HOME' HF_DATASETS_CACHE='$HF_DATASETS_CACHE' HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
  export PY='$PY' OUT_C='$OUT_C' OUT_D_QA='$OUT_D_QA' OUT_D_TOOL='$OUT_D_TOOL'
  export DATA_DIR='$DATA_DIR' RAW='$RAW'
  export FINEWEB_CHARS='$FINEWEB_CHARS' SYN_CHARS='$SYN_CHARS'
  export CONT_STEPS='$CONT_STEPS' QA_STEPS='$QA_STEPS' TOOL_STEPS='$TOOL_STEPS'
  export BENCH_LIMIT='$BENCH_LIMIT' BASE_CKPT='$BASE_CKPT'
  export QAT='$QAT' EMBED_QAT='$EMBED_QAT' SERVE_QUANT='$SERVE_QUANT'
  export PREP_MAX_CHARS='$PREP_MAX_CHARS' NEAR_DEDUP_J='$NEAR_DEDUP_J'
  export SKIP_A='${SKIP_A:-0}' SKIP_B='${SKIP_B:-0}' SKIP_C='${SKIP_C:-0}' SKIP_D='${SKIP_D:-0}'
  bash '$ROOT/scripts/run_spark1_dense_pipeline.sh'
  echo PIPE_EXIT=\$? \$(date -Iseconds) >>'$LOG'
"
sleep 2
echo "started screen $SCREEN_NAME on $(hostname)"
echo "  attach: screen -r $SCREEN_NAME"
echo "  log: $LOG"
screen -list || true
