#!/usr/bin/env bash
# Full capability ladder for C-headroom binary (~195M).
# base (mix_train) → multi-mix agent SFT → SciQ denser → export + SciQ@400 + battery.
#
#   setsid bash scripts/run_binary_c195_ladder.sh >>checkpoints/v78_binary_c195/ladder.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"

BASE_OUT="${BASE_OUT:-checkpoints/v78_binary_c195}"
SFT_OUT="${SFT_OUT:-checkpoints/v78_binary_c195_sft}"
SCIQ_OUT="${SCIQ_OUT:-checkpoints/v78_binary_c195_sciq}"
BASE_STEPS="${BASE_STEPS:-60000}"
SFT_EXTRA="${SFT_EXTRA:-12000}"
SCIQ_EXTRA="${SCIQ_EXTRA:-15000}"
SFT_DATA="${SFT_DATA:-data/cap_mix_v15_agent.txt}"
SCIQ_DATA="${SCIQ_DATA:-data/cap_mix_v16_binary_sciq.txt}"
BS="${BS:-4}"
ACCUM_BASE="${ACCUM_BASE:-16}"
ACCUM_SFT="${ACCUM_SFT:-8}"
GPU_FRAC="${GPU_FRAC:-0.55}"

log() { echo "[$(date -Is)] $*"; }

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "need ROCm torch" >&2
  exit 1
fi

mkdir -p "$BASE_OUT" "$SFT_OUT" "$SCIQ_OUT"
log "=== v78 binary C195 ladder start ==="
log "base=$BASE_OUT steps=$BASE_STEPS  sft=$SFT_OUT +$SFT_EXTRA  sciq=$SCIQ_OUT +$SCIQ_EXTRA"

# --- Stage 0: base pretrain (skip if best already at target steps) ---
need_base=1
if [[ -f "$BASE_OUT/best.pt" && -f "$BASE_OUT/progress.json" ]]; then
  cur=$("$PY" -c "import json; print(json.load(open('$BASE_OUT/progress.json'))['step'])")
  if (( cur >= BASE_STEPS )); then
    log "base already at step $cur >= $BASE_STEPS; skip"
    need_base=0
  fi
fi
if (( need_base )); then
  log "STAGE base: native binary L34×d576 mid2880"
  OUT="$BASE_OUT" STEPS="$BASE_STEPS" BS="$BS" ACCUM="$ACCUM_BASE" GPU_FRAC="$GPU_FRAC" \
    bash scripts/train_dense_binary_c195.sh
  log "base done"
fi

# free last if best exists (disk)
if [[ -f "$BASE_OUT/best.pt" && -f "$BASE_OUT/last.pt" ]]; then
  # keep last for resume chain; hardlink if identical size after finish
  :
fi

pick_ckpt() {
  local d="$1"
  if [[ -f "$d/best.pt" ]]; then echo "$d/best.pt"
  elif [[ -f "$d/last.pt" ]]; then echo "$d/last.pt"
  else echo ""; fi
}

# --- Stage 1: multi-mix agent SFT ---
CKPT=$(pick_ckpt "$BASE_OUT")
[[ -n "$CKPT" ]] || { log "no base ckpt"; exit 1; }
need_sft=1
if [[ -f "$SFT_OUT/progress.json" ]]; then
  bstep=$("$PY" -c "import torch; print(torch.load('$CKPT',map_location='cpu',weights_only=False)['step'])")
  sstep=$("$PY" -c "import json; print(json.load(open('$SFT_OUT/progress.json'))['step'])")
  if (( sstep >= bstep + SFT_EXTRA )); then
    log "sft already complete step $sstep; skip"
    need_sft=0
  fi
fi
if (( need_sft )); then
  log "STAGE multi-mix SFT from $CKPT data=$SFT_DATA"
  STEP=$("$PY" -c "import torch; print(torch.load('$CKPT',map_location='cpu',weights_only=False)['step'])")
  END=$((STEP + SFT_EXTRA))
  "$PY" -u -m ll3m.train \
    --resume "$CKPT" \
    --data "$SFT_DATA" --tokenizer data/tokenizer.json \
    --out-dir "$SFT_OUT" --steps "$END" \
    --batch-size "$BS" --grad-accum "$ACCUM_SFT" \
    --gpu-mem-fraction "$GPU_FRAC" \
    --lr 5e-6 --warmup 200 \
    --dropout 0.1 --weight-decay 0.1 \
    --device rocm --force \
    --qat binary --embed-qat int8 --serve-quant binary \
    --infer-scope ccd --patience 0 \
    --prompt "Once upon a time"
  log "sft done"
fi

# --- Stage 2: denser SciQ continue ---
CKPT=$(pick_ckpt "$SFT_OUT")
[[ -n "$CKPT" ]] || CKPT=$(pick_ckpt "$BASE_OUT")
need_sciq=1
if [[ -f "$SCIQ_OUT/progress.json" ]]; then
  bstep=$("$PY" -c "import torch; print(torch.load('$CKPT',map_location='cpu',weights_only=False)['step'])")
  sstep=$("$PY" -c "import json; print(json.load(open('$SCIQ_OUT/progress.json'))['step'])")
  if (( sstep >= bstep + SCIQ_EXTRA )); then
    log "sciq already complete step $sstep; skip"
    need_sciq=0
  fi
fi
if (( need_sciq )); then
  log "STAGE SciQ denser from $CKPT data=$SCIQ_DATA"
  STEP=$("$PY" -c "import torch; print(torch.load('$CKPT',map_location='cpu',weights_only=False)['step'])")
  END=$((STEP + SCIQ_EXTRA))
  "$PY" -u -m ll3m.train \
    --resume "$CKPT" \
    --data "$SCIQ_DATA" --tokenizer data/tokenizer.json \
    --out-dir "$SCIQ_OUT" --steps "$END" \
    --batch-size "$BS" --grad-accum "$ACCUM_SFT" \
    --gpu-mem-fraction "$GPU_FRAC" \
    --lr 5e-6 --warmup 200 \
    --dropout 0.1 --weight-decay 0.1 \
    --device rocm --force \
    --qat binary --embed-qat int8 --serve-quant binary \
    --infer-scope ccd --patience 0 \
    --prompt "Once upon a time"
  log "sciq done"
fi

# --- Stage 3: eval pack ---
CKPT=$(pick_ckpt "$SCIQ_OUT")
[[ -n "$CKPT" ]] || CKPT=$(pick_ckpt "$SFT_OUT")
[[ -n "$CKPT" ]] || { log "no ckpt for eval"; exit 1; }
EVAL_OUT="$SCIQ_OUT"
log "STAGE eval on $CKPT"

# CPU SciQ so VRAM stays free for export if needed
HIP_VISIBLE_DEVICES= "$PY" scripts/run_std_benchmarks.py \
  --checkpoint "$CKPT" --tokenizer data/tokenizer.json \
  --tasks sciq --limit 400 --device cpu \
  --out "$EVAL_OUT/std_benchmarks_sciq400.json" || true

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
"$PY" scripts/export_weights.py \
  --checkpoint "$CKPT" --out "$EVAL_OUT/best.ll3m" --quant binary

# C residency smoke
if [[ -x ./c_infer/ll3m_c ]]; then
  ./c_infer/ll3m_c -m "$EVAL_OUT/best.ll3m" --token-ids 1,2,3 -n 4 --warm 2 --ccd 0 \
    | tee "$EVAL_OUT/c_smoke.log" || true
fi

"$PY" scripts/eval_agent_battery.py \
  --checkpoint "$EVAL_OUT/best.ll3m" --no-assist --temperature 0 \
  --out "$EVAL_OUT/battery_product_gate_on.json" || true

"$PY" - <<PY
import json
from pathlib import Path
o = Path("$EVAL_OUT")
sq = None
p = o / "std_benchmarks_sciq400.json"
if p.exists():
    sq = json.loads(p.read_text())["tasks"]["sciq"]["acc_pct"]
bat = None
bp = o / "battery_product_gate_on.json"
if bp.exists():
    try:
        j = json.loads(bp.read_text())
        bat = j.get("pass_n") or j.get("passed") or j.get("summary")
    except Exception as e:
        bat = str(e)
lines = [
    "v78 binary C195 ladder",
    f"ckpt=$CKPT",
    f"SciQ@400={sq}",
    f"battery={bat}",
    "shape=L34×d576 mid2880 ~194.9M binary+int8",
]
(o / "DECISION.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

log "=== ladder complete ==="
