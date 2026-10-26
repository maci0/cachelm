#!/usr/bin/env bash
# From LOCAL machine: rsync code to spark2, start pipeline in remote screen.
# After this returns, local machine can reboot freely.
#
#   bash scripts/launch_spark2_base_en.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST="${SPARK2_HOST:-maci@192.168.0.212}"
KEY="${SPARK2_KEY:-$HOME/.config/NVIDIA/Sync/config/nvsync.key}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/maci/Desktop/cachelm}"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=15 "$HOST")
RSYNC_SSH="ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=15"

echo "=== rsync code → $HOST:$REMOTE_ROOT (no local checkpoints/data required) ==="
rsync -az --delete \
  -e "$RSYNC_SSH" \
  --exclude '.venv/' \
  --exclude 'checkpoints/' \
  --exclude 'data/mix_*' \
  --exclude 'data/TinyStories*' \
  --exclude 'data/tool_sft*' \
  --exclude 'data/base_en/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude 'graphify-out/' \
  "$ROOT/" "$HOST:$REMOTE_ROOT/"

echo "=== remote start pipeline (screen ll3m-base-en) ==="
"${SSH[@]}" "bash -s" <<EOS
set -euo pipefail
cd $REMOTE_ROOT
# ensure screen
command -v screen >/dev/null
# if already running, do not clobber
if screen -list 2>/dev/null | grep -q '[.]ll3m-base-en[[:space:]]'; then
  echo "ALREADY_RUNNING screen ll3m-base-en"
  screen -list
  cat checkpoints/v76_base_en/progress.json 2>/dev/null || true
  tail -5 checkpoints/v76_base_en/pipeline.log 2>/dev/null || true
  exit 0
fi
mkdir -p checkpoints/v76_base_en
FINEWEB_CHARS=${FINEWEB_CHARS:-1500000000} \
SYN_CHARS=${SYN_CHARS:-250000000} \
STORIES_CHARS=${STORIES_CHARS:-0} \
STEPS=${STEPS:-60000} \
SCREEN_NAME=ll3m-base-en \
  bash scripts/run_v76_base_en_pipeline_cuda.sh
EOS

echo
echo "=== local status snapshot ==="
bash "$ROOT/scripts/poll_spark2_base_en.sh" || true

echo
echo "DONE. Training/data pipeline is on spark2 only."
echo "  Local reboot: safe. Job is screen ll3m-base-en on $HOST"
echo "  Status once:  bash scripts/poll_spark2_base_en.sh"
echo "  Status loop:  bash scripts/poll_spark2_base_en.sh --watch"
echo "  Attach:       ssh -i $KEY -o IdentitiesOnly=yes $HOST -t screen -r ll3m-base-en"
echo "  Local mirror: checkpoints/v76_base_en_status/latest.txt"
