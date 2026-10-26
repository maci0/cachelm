#!/usr/bin/env python3
"""Aggregate Qwen3.5-* int4 MC JSONs and diff vs LL3M denser checkpoints.

  uv run python scripts/aggregate_qwen_family_vs_ll3m.py \\
    --compare-dir checkpoints/compare --limit 400
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _task_acc(blob: dict, task: str) -> float | None:
    t = blob.get("tasks", {}).get(task)
    if isinstance(t, dict) and "acc_pct" in t:
        return float(t["acc_pct"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare-dir", type=str, default="checkpoints/compare")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument(
        "--out",
        type=str,
        default="checkpoints/compare/ll3m_vs_qwen35_family_int4.json",
    )
    ap.add_argument(
        "--ll3m",
        type=str,
        nargs="*",
        default=[
            "checkpoints/v76_dense_c/vs_qwen4_expC.json",
            "checkpoints/v76_dense_c/std_benchmarks_expC_limit400.json",
            "checkpoints/v76_dense_mix/vs_qwen4_expMix.json",
            "checkpoints/v76_dense_mix/std_benchmarks_expMix_limit400.json",
            "checkpoints/v76_dense_tool/vs_qwen4_expD.json",
            "checkpoints/v76_dense_tool/std_benchmarks_expD_limit400.json",
        ],
    )
    args = ap.parse_args()
    cdir = Path(args.compare_dir)

    # Prefer full std_benchmarks for LL3M (has tasks)
    ll3m: dict[str, dict] = {}
    for p in args.ll3m:
        path = Path(p)
        if not path.is_file():
            continue
        d = _load(path)
        if not d:
            continue
        # std_benchmarks style
        if "tasks" in d and "macro_acc_pct" in d:
            tag = path.stem
            if "expC" in tag or "dense_c" in str(path):
                key = "ll3m_C"
            elif "expMix" in tag or "dense_mix" in str(path):
                key = "ll3m_MIX"
            elif "expD" in tag or "dense_tool" in str(path):
                key = "ll3m_D"
            else:
                key = tag
            ll3m[key] = {
                "path": str(path),
                "macro_acc_pct": d["macro_acc_pct"],
                "tasks": {
                    t: v["acc_pct"]
                    for t, v in d["tasks"].items()
                    if isinstance(v, dict) and "acc_pct" in v
                },
            }
        # vs_qwen style (rows)
        elif "rows" in d and "ours_macro" in d:
            if "expC" in path.name:
                key = "ll3m_C"
            elif "expMix" in path.name:
                key = "ll3m_MIX"
            elif "expD" in path.name:
                key = "ll3m_D"
            else:
                key = path.stem
            if key in ll3m:
                continue
            ll3m[key] = {
                "path": str(path),
                "macro_acc_pct": d["ours_macro"],
                "tasks": {
                    r["task"]: r["ours_pct"]
                    for r in d["rows"]
                    if r.get("ours_pct") is not None
                },
            }

    # Also try local std_benchmarks under v76 dirs
    root = Path(__file__).resolve().parent.parent
    for tag, rel in [
        ("ll3m_C", "checkpoints/v76_dense_c/std_benchmarks_expC_limit400.json"),
        ("ll3m_MIX", "checkpoints/v76_dense_mix/std_benchmarks_expMix_limit400.json"),
        ("ll3m_D", "checkpoints/v76_dense_tool/std_benchmarks_expD_limit400.json"),
    ]:
        p = root / rel
        d = _load(p)
        if d and "macro_acc_pct" in d:
            ll3m[tag] = {
                "path": str(p),
                "macro_acc_pct": d["macro_acc_pct"],
                "tasks": {
                    t: v["acc_pct"]
                    for t, v in d.get("tasks", {}).items()
                    if isinstance(v, dict) and "acc_pct" in v
                },
            }

    qwen_models: list[dict] = []
    # Only portable weight-only int4 (reject NF4 / bitsandbytes / NVFP4 leftovers).
    allowed_modes = {"weight_only_int4_absmax"}
    for path in sorted(cdir.glob(f"*int4_limit{args.limit}.json")):
        name = path.name
        if "ll3m" in name.lower():
            continue
        if "nf4" in name.lower() or "nvfp4" in name.lower():
            continue
        d = _load(path)
        if not d or "macro_acc_pct" not in d:
            continue
        mode = (d.get("quant") or {}).get("mode")
        if mode not in allowed_modes:
            print(f"skip {path.name}: quant.mode={mode!r} (want weight_only_int4_absmax)")
            continue
        model = d.get("model") or name
        if "Qwen" not in str(model) and "Qwen" not in name:
            continue
        qwen_models.append(
            {
                "model": model,
                "path": str(path),
                "n_params": d.get("n_params"),
                "quant": d.get("quant"),
                "macro_acc_pct": d["macro_acc_pct"],
                "tasks": {
                    t: v["acc_pct"]
                    for t, v in d.get("tasks", {}).items()
                    if isinstance(v, dict) and "acc_pct" in v
                },
            }
        )

    # Sort by params if known
    qwen_models.sort(key=lambda x: (x.get("n_params") or 0, str(x["model"])))

    # Build comparison rows vs each LL3M
    comparisons: dict[str, list] = {}
    for lkey, lblob in ll3m.items():
        rows = []
        for q in qwen_models:
            delta = lblob["macro_acc_pct"] - q["macro_acc_pct"]
            rel = (
                100.0 * (1.0 - lblob["macro_acc_pct"] / q["macro_acc_pct"])
                if q["macro_acc_pct"]
                else None
            )
            task_deltas = {}
            for t, oa in lblob.get("tasks", {}).items():
                qa = q["tasks"].get(t)
                if qa is not None:
                    task_deltas[t] = {
                        "ll3m_pct": oa,
                        "qwen_pct": qa,
                        "delta_pp": oa - qa,
                    }
            rows.append(
                {
                    "qwen_model": q["model"],
                    "qwen_n_params": q.get("n_params"),
                    "qwen_macro": q["macro_acc_pct"],
                    "ll3m_macro": lblob["macro_acc_pct"],
                    "delta_pp": delta,
                    "ll3m_relative_deficit_pct": rel,
                    "param_ratio_qwen_over_ll3m": (
                        q["n_params"] / 78_815_712 if q.get("n_params") else None
                    ),
                    "tasks": task_deltas,
                }
            )
        comparisons[lkey] = rows

    report = {
        "protocol": (
            f"zero-shot continuation log-likelihood MC, limit={args.limit}, seed=42, "
            "Qwen weight-only per-row absmax int4 (not NF4/NVFP4)"
        ),
        "ll3m_params": 78_815_712,
        "ll3m": ll3m,
        "qwen_family": qwen_models,
        "comparisons": comparisons,
        "macro_table": {
            "ll3m": {k: v["macro_acc_pct"] for k, v in ll3m.items()},
            "qwen": [
                {
                    "model": q["model"],
                    "n_params": q.get("n_params"),
                    "macro": q["macro_acc_pct"],
                }
                for q in qwen_models
            ],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== macro table ===")
    for k, v in ll3m.items():
        print(f"  {k:12} {v['macro_acc_pct']:.2f}%")
    for q in qwen_models:
        np = q.get("n_params")
        np_s = f"{np/1e9:.2f}B" if np else "?"
        print(f"  {q['model']:40} {np_s:>8}  macro={q['macro_acc_pct']:.2f}%")
    if "ll3m_C" in comparisons:
        print("=== LL3M C vs family (delta_pp) ===")
        for r in comparisons["ll3m_C"]:
            print(
                f"  vs {r['qwen_model']}: Δ={r['delta_pp']:+.2f} pp  "
                f"rel_deficit={r['ll3m_relative_deficit_pct']:.1f}%  "
                f"params×{r['param_ratio_qwen_over_ll3m']:.1f}"
                if r.get("ll3m_relative_deficit_pct") is not None
                and r.get("param_ratio_qwen_over_ll3m") is not None
                else f"  vs {r['qwen_model']}: Δ={r['delta_pp']:+.2f} pp"
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
