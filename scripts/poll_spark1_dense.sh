#!/usr/bin/env bash
# Optional LOCAL status mirror for spark1 dense ladder (A→D).
# Safe to kill anytime (local only; pipeline lives in screen ll3m-dense).
#
#   bash scripts/poll_spark1_dense.sh          # once
#   bash scripts/poll_spark1_dense.sh --watch  # loop every 120s
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST="${SPARK1_HOST:-maci@192.168.0.211}"
KEY="${SPARK1_KEY:-$HOME/.config/NVIDIA/Sync/config/nvsync.key}"
REMOTE="${REMOTE_ROOT:-/home/maci/Desktop/cachelm}"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=15 -o BatchMode=yes "$HOST")
LOCAL_STATUS=checkpoints/dense_pipeline
mkdir -p "$LOCAL_STATUS"
INTERVAL="${POLL_INTERVAL:-120}"
WATCH=0
[[ "${1:-}" == "--watch" ]] && WATCH=1

once() {
  echo "==== spark1 dense $(date -Iseconds) ===="
  "${SSH[@]}" "bash -s" <<'EOS' | tee "$LOCAL_STATUS/latest.txt" || echo "SSH_FAILED=1"
set -euo pipefail
cd ~/Desktop/cachelm
echo "host=$(hostname)"
screen -list 2>/dev/null | grep -E 'll3m-dense|No Sockets' || true
pgrep -af 'download_base|prepare_base|ll3m.train|validate_vs|run_spark1' 2>/dev/null | grep -v pgrep | head -8 || echo "no_train_procs"
echo "--- sizes ---"
ls -lh data/base_en_dense/raw/* 2>/dev/null || echo "no raw yet"
ls -lh data/base_en_dense/mix_train.bin data/base_en_dense/manifest.json 2>/dev/null || true
for d in v76_dense_c v76_dense_qa v76_dense_tool; do
  if [[ -f checkpoints/$d/progress.json ]]; then
    echo -n "$d progress: "
    cat checkpoints/$d/progress.json
    echo
  fi
  if [[ -f checkpoints/$d/best.pt ]]; then
    ls -lh checkpoints/$d/best.pt checkpoints/$d/vs_qwen4_*.json 2>/dev/null || ls -lh checkpoints/$d/best.pt
  fi
done
echo "--- log tail ---"
tail -25 checkpoints/dense_pipeline/pipeline.log 2>/dev/null || true
if [[ -f checkpoints/dense_pipeline/DONE.txt ]]; then
  echo "DONE: $(cat checkpoints/dense_pipeline/DONE.txt)"
fi
EOS
  # pull compare results when present
  rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=15" \
    "$HOST:$REMOTE/checkpoints/v76_dense_c/vs_qwen4_*.json" \
    "$HOST:$REMOTE/checkpoints/v76_dense_c/std_benchmarks_*.json" \
    "$LOCAL_STATUS/" 2>/dev/null || true
  rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=15" \
    "$HOST:$REMOTE/checkpoints/v76_dense_tool/vs_qwen4_*.json" \
    "$HOST:$REMOTE/checkpoints/v76_dense_tool/std_benchmarks_*.json" \
    "$LOCAL_STATUS/" 2>/dev/null || true
  rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=15" \
    "$HOST:$REMOTE/checkpoints/dense_pipeline/pipeline.log" \
    "$LOCAL_STATUS/pipeline.log" 2>/dev/null || true
}

if [[ "$WATCH" -eq 1 ]]; then
  echo "watching every ${INTERVAL}s (Ctrl-C safe; does not stop spark1 train)"
  while true; do
    once || true
    sleep "$INTERVAL"
  done
else
  once
fi
