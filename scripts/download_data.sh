#!/usr/bin/env bash
# Download / expand TinyStories training data for ll3m.
# Prefers GPT-4-only V2 (higher quality) over the mixed GPT-3.5/4 original.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/data"
mkdir -p "$DATA"
cd "$DATA"

HF="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

# Bytes to fetch for GPT-4 train (full file is ~2.23 GiB).
# Default ~1.5 GiB for 75M-scale train; MAX_BYTES=0 for full file.
MAX_BYTES="${MAX_BYTES:-1610612736}"

fetch_range() {
  local url="$1" out="$2" bytes="$3"
  if [[ -f "$out" ]]; then
    local have
    have=$(wc -c <"$out")
    if [[ "$bytes" -eq 0 ]]; then
      echo "exists $out ($have bytes); set FORCE=1 to re-download"
      [[ "${FORCE:-0}" == "1" ]] || return 0
    elif [[ "$have" -ge "$bytes" ]]; then
      echo "exists $out ($have bytes >= $bytes); skip"
      return 0
    fi
  fi
  if [[ "$bytes" -eq 0 ]]; then
    echo "downloading full → $out"
    curl -L --fail --retry 3 -o "$out.partial" "$url"
    mv "$out.partial" "$out"
  else
    local end=$((bytes - 1))
    echo "downloading bytes 0-${end} → $out"
    curl -L --fail --retry 3 -r "0-${end}" -o "$out.partial" "$url"
    mv "$out.partial" "$out"
  fi
  wc -c "$out"
}

echo "=== TinyStoriesV2 GPT-4 train (quality) ==="
if [[ "$MAX_BYTES" -eq 0 ]]; then
  fetch_range "${HF}/TinyStoriesV2-GPT4-train.txt" \
    "TinyStoriesV2-GPT4-train.txt" 0
else
  fetch_range "${HF}/TinyStoriesV2-GPT4-train.txt" \
    "TinyStoriesV2-GPT4-train.partial.txt" "$MAX_BYTES"
fi

echo "=== TinyStoriesV2 GPT-4 valid (official holdout) ==="
fetch_range "${HF}/TinyStoriesV2-GPT4-valid.txt" \
  "TinyStoriesV2-GPT4-valid.txt" 0

# Shakespeare (small, already usually present)
if [[ ! -f tinyshakespeare.txt ]]; then
  echo "=== Tiny Shakespeare ==="
  curl -L --fail -o tinyshakespeare.txt \
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
fi

echo
echo "done. Build mix + BPE with:"
echo "  uv run python scripts/prepare_data.py --vocab-size 2048"
echo "Sources preferred automatically: GPT-4 train (+ official valid if present)."
