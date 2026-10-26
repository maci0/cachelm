#!/usr/bin/env python3
"""Run std MC benchmarks for a .pt checkpoint and diff vs Qwen3.5-0.8B int4 baseline.

  uv run python scripts/validate_vs_qwen4.py \\
    --checkpoint checkpoints/v76_dense_c/best.pt \\
    --tag expC \\
    --limit 400
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Portable weight-only int4 (not NF4). Prefer Qwen__* naming from family script.
QWEN_BASELINE = ROOT / "checkpoints" / "compare" / "Qwen__Qwen3.5-0.8B-Base_int4_limit400.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="data/base_en/tokenizer.json")
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    out_dir = ck.parent
    out_json = out_dir / f"std_benchmarks_{args.tag}_limit{args.limit}.json"
    cmp_json = out_dir / f"vs_qwen4_{args.tag}.json"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_std_benchmarks.py"),
        "--checkpoint",
        str(ck),
        "--tokenizer",
        args.tokenizer,
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--out",
        str(out_json),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))

    ours = json.loads(out_json.read_text())
    qwen = {}
    if QWEN_BASELINE.is_file():
        qwen = json.loads(QWEN_BASELINE.read_text())

    rows = []
    for task, met in ours.get("tasks", {}).items():
        if not isinstance(met, dict) or "acc_pct" not in met:
            continue
        q = qwen.get("tasks", {}).get(task, {})
        rows.append(
            {
                "task": task,
                "ours_pct": met["acc_pct"],
                "qwen4_pct": q.get("acc_pct"),
                "random_pct": met.get("random_pct"),
                "delta_vs_qwen": (
                    met["acc_pct"] - q["acc_pct"]
                    if isinstance(q, dict) and "acc_pct" in q
                    else None
                ),
                "n": met.get("n"),
            }
        )

    report = {
        "tag": args.tag,
        "checkpoint": str(ck),
        "ours_macro": ours.get("macro_acc_pct"),
        "qwen4_macro": qwen.get("macro_acc_pct"),
        "qwen_baseline": str(QWEN_BASELINE) if qwen else None,
        "rows": rows,
    }
    cmp_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {cmp_json}")


if __name__ == "__main__":
    main()
