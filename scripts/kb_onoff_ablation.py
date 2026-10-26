#!/usr/bin/env python3
"""KB on/off ablation for agent battery lookup (+ optional pure/product).

  uv run python scripts/kb_onoff_ablation.py \
    --checkpoint checkpoints/v76_dense_c_pure_v2/best.ll3m \
    --out checkpoints/v76_dense_c_pure_v2/kb_onoff_ablation.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ll3m.chat import LL3M
from ll3m.tools import kb_size, set_kb
from scripts.eval_agent_battery import build_agent_cases, run_case


def _score_lookup_subset(m: LL3M, *, gate: bool, assist: bool = False) -> dict:
    cases = [(n, c, u, kw) for n, c, u, kw in build_agent_cases() if c == "lookup"]
    results = []
    t0 = time.perf_counter()
    for name, cat, user, kw in cases:
        r = run_case(
            m,
            name,
            cat,
            user,
            assist=assist,
            gate_tools=gate,
            temperature=0.0,
            top_k=1,
            **kw,
        )
        results.append(
            {
                "name": r.name,
                "ok": r.ok,
                "detail": r.detail,
                "final": r.final[:120],
                "executed_tools": r.executed_tools,
                "model_emitted_call": r.model_emitted_call,
            }
        )
    n_ok = sum(1 for r in results if r["ok"])
    return {
        "n": len(results),
        "passed": n_ok,
        "pass_rate": n_ok / max(len(results), 1),
        "elapsed_s": time.perf_counter() - t0,
        "kb_size": kb_size(),
        "failures": [r for r in results if not r["ok"]],
        "all": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="checkpoints/v76_dense_c_pure_v2/best.ll3m",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--python", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"missing {ck}")

    m = LL3M(ck, python_infer=args.python, seed=args.seed)

    # KB ON (default load)
    set_kb(None)
    print(f"=== KB ON size={kb_size()} product gate ===", flush=True)
    on_prod = _score_lookup_subset(m, gate=True)
    print(f"  lookup {on_prod['passed']}/{on_prod['n']}", flush=True)

    print(f"=== KB ON size={kb_size()} pure ===", flush=True)
    on_pure = _score_lookup_subset(m, gate=False)
    print(f"  lookup {on_pure['passed']}/{on_pure['n']}", flush=True)

    # KB OFF
    set_kb({})
    print(f"=== KB OFF size={kb_size()} product gate ===", flush=True)
    off_prod = _score_lookup_subset(m, gate=True)
    print(f"  lookup {off_prod['passed']}/{off_prod['n']}", flush=True)

    print(f"=== KB OFF size={kb_size()} pure ===", flush=True)
    off_pure = _score_lookup_subset(m, gate=False)
    print(f"  lookup {off_pure['passed']}/{off_pure['n']}", flush=True)

    # restore
    set_kb(None)

    report = {
        "checkpoint": str(ck),
        "seed": args.seed,
        "protocol": "lookup subset only (20 cases); greedy temp=0 top_k=1",
        "kb_on": {"product_gate": on_prod, "pure": on_pure},
        "kb_off": {"product_gate": off_prod, "pure": off_pure},
        "delta_product_pp": (on_prod["pass_rate"] - off_prod["pass_rate"]) * 100,
        "delta_pure_pp": (on_pure["pass_rate"] - off_pure["pass_rate"]) * 100,
        "interpretation": (
            "If lookup accuracy collapses with empty KB, hits are host-memory "
            "not weight knowledge. Gate should not invent lookups."
        ),
    }
    out = Path(
        args.out
        or (ck.parent / "kb_onoff_ablation.json")
    )
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k not in ("kb_on", "kb_off")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
