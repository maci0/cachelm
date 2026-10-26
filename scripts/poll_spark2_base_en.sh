#!/usr/bin/env bash
# Optional LOCAL status mirror for spark2 v76 base-English train.
# Safe to kill anytime (local reboot kills this only; train continues on spark2).
#
#   bash scripts/poll_spark2_base_en.sh          # once
#   bash scripts/poll_spark2_base_en.sh --watch  # loop every 120s
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST="${SPARK2_HOST:-maci@192.168.0.212}"
KEY="${SPARK2_KEY:-$HOME/.config/NVIDIA/Sync/config/nvsync.key}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/maci/Desktop/cachelm}"
REMOTE_OUT="${REMOTE_OUT:-$REMOTE_ROOT/checkpoints/v76_base_en}"
LOCAL_STATUS="${LOCAL_STATUS:-$ROOT/checkpoints/v76_base_en_status}"
WATCH=0
INTERVAL="${INTERVAL:-120}"

for a in "$@"; do
  case "$a" in
    --watch) WATCH=1 ;;
    --interval=*) INTERVAL="${a#*=}" ;;
  esac
done

SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=15 -o BatchMode=yes "$HOST")

mkdir -p "$LOCAL_STATUS"

snapshot() {
  local ts
  ts=$(date -Iseconds)
  mkdir -p "$LOCAL_STATUS"
  {
    echo "polled_at=$ts"
    echo "remote=$HOST:$REMOTE_OUT"
    echo "note=local poller is optional; train lives only on spark2 screen ll3m-base-en"
    echo
    "${SSH[@]}" "bash -s" <<'EOS' || echo "SSH_FAILED=1"
set +e
ROOT=/home/maci/Desktop/cachelm
OUT=$ROOT/checkpoints/v76_base_en
echo "hostname=$(hostname)"
echo "date=$(date -Iseconds)"
echo "screen:"
screen -list 2>/dev/null | sed 's/^/  /' || true
echo "train_procs:"
pgrep -af 'll3m.train|run_v76_base_en|download_base_english|prepare_base_english' 2>/dev/null | sed 's/^/  /' || echo "  (none)"
echo
if [[ -f $OUT/progress.json ]]; then
  echo "=== progress.json ==="
  cat $OUT/progress.json
  echo
fi
if [[ -f $OUT/pipeline.log ]]; then
  echo "=== pipeline.log tail ==="
  tail -20 $OUT/pipeline.log
  echo
fi
if [[ -f $OUT/train.log ]]; then
  echo "=== train.log tail ==="
  tail -15 $OUT/train.log
  echo
fi
if [[ -f $OUT/RUN_ON_SPARK2.txt ]]; then
  echo "=== RUN_ON_SPARK2.txt ==="
  cat $OUT/RUN_ON_SPARK2.txt
fi
if [[ -f $ROOT/data/base_en/manifest.json ]]; then
  echo "=== data manifest (summary) ==="
  python3 -c "import json;d=json.load(open('$ROOT/data/base_en/manifest.json'));print({k:d[k] for k in d if k in ('train_tokens','val_tokens','train_chars','train_docs','goal','vocab_size','sources')})" 2>/dev/null || true
fi
ls -lah $OUT 2>/dev/null | head -20
EOS
  } >"$LOCAL_STATUS/latest.txt"

  # compact JSON if progress available
  if grep -q '"step"' "$LOCAL_STATUS/latest.txt" 2>/dev/null; then
    "${SSH[@]}" "cat $REMOTE_OUT/progress.json" >"$LOCAL_STATUS/progress.json" 2>/dev/null || true
  fi
  cp "$LOCAL_STATUS/latest.txt" "$LOCAL_STATUS/latest_${ts//[:]/}.txt" 2>/dev/null || true
  # keep last 30 snapshots
  ls -1t "$LOCAL_STATUS"/latest_*.txt 2>/dev/null | tail -n +31 | xargs -r rm -f
  echo "wrote $LOCAL_STATUS/latest.txt"
  # short console summary
  rg -n 'step|best_val|SCREEN|train_procs|SSH_FAILED|pipeline|done' "$LOCAL_STATUS/latest.txt" | head -25 || head -40 "$LOCAL_STATUS/latest.txt"
}

if [[ "$WATCH" -eq 1 ]]; then
  echo "watching every ${INTERVAL}s (Ctrl-C safe; does not stop spark2 train)"
  while true; do
    snapshot || true
    sleep "$INTERVAL"
  done
else
  snapshot
fi
