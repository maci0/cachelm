#!/usr/bin/env bash
# Eval a train out-dir for capability loop: export L3T2 + product battery + optional SciQ.
# Usage:
#   bash scripts/eval_candidate.sh checkpoints/v76_dense_c_cap1
#   SCIQ=1 bash scripts/eval_candidate.sh checkpoints/v76_dense_c_cap1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DIR="${1:?out-dir}"
PY="${PY:-$ROOT/.venv/bin/python}"
CKPT="$DIR/best.pt"
[[ -f "$CKPT" ]] || CKPT="$DIR/last.pt"
[[ -f "$CKPT" ]] || { echo "no best.pt/last.pt in $DIR" >&2; exit 1; }

LL3M="$DIR/best.ll3m"
echo "=== export ternary $CKPT → $LL3M ==="
"$PY" scripts/export_weights.py --checkpoint "$CKPT" --out "$LL3M" --quant ternary

echo "=== product battery (assist off, gate default) ==="
"$PY" scripts/eval_agent_battery.py \
  --checkpoint "$LL3M" --no-assist --temperature 0 \
  --out "$DIR/battery_product_gate_on.json"

if [[ "${SCIQ:-0}" == "1" ]]; then
  echo "=== SciQ limit 400 ==="
  "$PY" scripts/run_std_benchmarks.py \
    --checkpoint "$CKPT" --tasks sciq --limit 400 \
    --out "$DIR/std_benchmarks_sciq400.json" || true
fi

"$PY" - <<PY
import json
from pathlib import Path
d = Path("$DIR")
bat = json.loads((d / "battery_product_gate_on.json").read_text())
print(f"battery {bat.get('passed')}/{bat.get('total')}  rate={bat.get('pass_rate')}")
print("by", bat.get("by_category"))
sciq = d / "std_benchmarks_sciq400.json"
if sciq.is_file():
    s = json.loads(sciq.read_text())
    print("sciq", s)
print("pack", "$LL3M")
PY
