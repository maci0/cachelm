#!/usr/bin/env bash
# From LOCAL: rsync to spark1 and start dense knowledge pipeline in screen.
# After return, local reboot is safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST="${SPARK1_HOST:-maci@192.168.0.211}"
KEY="${SPARK1_KEY:-$HOME/.config/NVIDIA/Sync/config/nvsync.key}"
REMOTE="${REMOTE_ROOT:-/home/maci/Desktop/cachelm}"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=20 "$HOST")
RSYNC_SSH="ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=20"

echo "=== rsync code → $HOST:$REMOTE ==="
rsync -az --delete \
  -e "$RSYNC_SSH" \
  --exclude '.venv/' \
  --exclude 'checkpoints/' \
  --exclude 'data/mix_*' \
  --exclude 'data/TinyStories*' \
  --exclude 'data/base_en/raw/' \
  --exclude 'data/base_en_dense/' \
  --exclude '.cache/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude 'graphify-out/' \
  "$ROOT/" "$HOST:$REMOTE/"

echo "=== rsync base_en tokenizer + v76 best.pt from local if present ==="
rsync -az -e "$RSYNC_SSH" \
  "$ROOT/data/base_en/tokenizer.json" \
  "$ROOT/data/base_en/manifest.json" \
  "$HOST:$REMOTE/data/base_en/" 2>/dev/null || true
# Qwen3.5 weight-only int4 baselines for post-C/D validate (checkpoints/ excluded above)
if [[ -f "$ROOT/checkpoints/compare/Qwen__Qwen3.5-0.8B-Base_int4_limit400.json" ]]; then
  "${SSH[@]}" "mkdir -p $REMOTE/checkpoints/compare"
  rsync -az -e "$RSYNC_SSH" \
    "$ROOT/checkpoints/compare/Qwen__Qwen3.5-0.8B-Base_int4_limit400.json" \
    "$ROOT/checkpoints/compare/ll3m_vs_qwen35_family_int4.json" \
    "$HOST:$REMOTE/checkpoints/compare/" || true
fi
if [[ -f "$ROOT/checkpoints/v76_base_en/best.pt" ]]; then
  "${SSH[@]}" "mkdir -p $REMOTE/checkpoints/v76_base_en"
  rsync -az -e "$RSYNC_SSH" \
    "$ROOT/checkpoints/v76_base_en/best.pt" \
    "$ROOT/checkpoints/v76_base_en/best.ll3m" \
    "$HOST:$REMOTE/checkpoints/v76_base_en/" || true
fi
# also try spark2 if local missing
if ! "${SSH[@]}" "test -f $REMOTE/checkpoints/v76_base_en/best.pt"; then
  echo "pulling v76 best.pt from spark2…"
  ssh -i "$KEY" -o IdentitiesOnly=yes maci@192.168.0.212 \
    "test -f ~/Desktop/cachelm/checkpoints/v76_base_en/best.pt" && \
  rsync -az -e "$RSYNC_SSH" \
    maci@192.168.0.212:~/Desktop/cachelm/checkpoints/v76_base_en/best.pt \
    "$HOST:$REMOTE/checkpoints/v76_base_en/" || true
fi

echo "=== bootstrap venv on spark1 if needed ==="
"${SSH[@]}" "bash -s" <<EOS
set -euo pipefail
cd $REMOTE
mkdir -p checkpoints/dense_pipeline data/base_en
if [[ ! -x .venv/bin/python ]]; then
  echo "creating venv…"
  if command -v uv >/dev/null; then
    uv venv .venv
  else
    python3 -m venv .venv
  fi
fi
# CUDA torch if missing
if ! .venv/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "installing torch cuda…"
  if command -v uv >/dev/null; then
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130 || \
    uv pip install --python .venv/bin/python torch
  else
    .venv/bin/pip install -U pip
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130 || \
    .venv/bin/pip install torch
  fi
fi
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# project deps
if command -v uv >/dev/null; then
  uv pip install --python .venv/bin/python numpy datasets huggingface_hub 2>/dev/null || true
else
  .venv/bin/pip install numpy datasets huggingface_hub 2>/dev/null || true
fi
# editable install / path
export PYTHONPATH=$REMOTE
.venv/bin/python -c "from ll3m.tools import _KB; print('KB', len(_KB))"
EOS

echo "=== start dense pipeline screen ==="
"${SSH[@]}" "bash -s" <<EOS
set -euo pipefail
cd $REMOTE
export PYTHONPATH=$REMOTE
if screen -list 2>/dev/null | grep -q '[.]ll3m-dense[[:space:]]'; then
  echo ALREADY_RUNNING
  screen -list
  tail -20 checkpoints/dense_pipeline/pipeline.log 2>/dev/null || true
  exit 0
fi
SYN_CHARS=\${SYN_CHARS:-700000000} FINEWEB_CHARS=\${FINEWEB_CHARS:-500000000} \
CONT_STEPS=\${CONT_STEPS:-30000} BENCH_LIMIT=\${BENCH_LIMIT:-400} \
QAT=\${QAT:-ternary} EMBED_QAT=\${EMBED_QAT:-int8} SERVE_QUANT=\${SERVE_QUANT:-ternary} \
PREP_MAX_CHARS=\${PREP_MAX_CHARS:-1200} NEAR_DEDUP_J=\${NEAR_DEDUP_J:-0.85} \
SKIP_A=\${SKIP_A:-0} SKIP_B=\${SKIP_B:-0} SKIP_C=\${SKIP_C:-0} SKIP_D=\${SKIP_D:-0} \
  bash scripts/run_spark1_dense_pipeline.sh
EOS

echo
echo "DONE. Pipeline on spark1 only (screen ll3m-dense)."
echo "  ssh -i $KEY -o IdentitiesOnly=yes $HOST -t screen -r ll3m-dense"
echo "  ssh … 'tail -f ~/Desktop/cachelm/checkpoints/dense_pipeline/pipeline.log'"
echo "  Local reboot: SAFE"
