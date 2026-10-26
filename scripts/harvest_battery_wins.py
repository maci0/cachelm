#!/usr/bin/env python3
"""Self-distill: convert successful agent-battery cases into tool SFT episodes.

GLM5.2 lesson: SFT on the model's *own* good trajectories preserves policy better
than imitating a foreign teacher. Use after no-assist battery JSON is written.

  uv run python scripts/harvest_battery_wins.py \\
    --battery checkpoints/v76_dense_c/battery_C_no_assist.json \\
    --out data/tool_sft_self_distill.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ll3m.agent import host_assist_call
from ll3m.tools import execute_call, format_result, parse_call, tools_brief


def _sys() -> str:
    return (
        "You are a small helper. Use tools for facts and math. "
        "Answer briefly in plain English.\n" + tools_brief()
    )


def _tool_names(tools: object) -> list[str]:
    if not tools:
        return []
    out: list[str] = []
    for t in tools:
        if isinstance(t, str):
            out.append(t.lower())
        elif isinstance(t, dict) and t.get("tool"):
            out.append(str(t["tool"]).lower())
    return out


def _episode_from_case(case: dict) -> str | None:
    if not case.get("ok"):
        return None
    cat = case.get("category") or ""
    if cat in ("host", "tokenizer", "language"):
        return None
    user = (case.get("user") or "").strip()
    final = (case.get("final") or "").strip()
    if not user or not final:
        return None
    names = _tool_names(case.get("tools"))
    tools_raw = case.get("tools") or []

    # Full dict trace (preferred)
    if tools_raw and isinstance(tools_raw[0], dict) and tools_raw[0].get("tool"):
        t0 = tools_raw[0]
        name = str(t0["tool"])
        args = t0.get("args") or []
        if isinstance(args, list):
            arg_s = " ".join(str(a) for a in args)
        else:
            arg_s = str(args)
        call = f"<call>{name} {arg_s}</call>".strip()
        call = re.sub(r"<call>(\w+)\s+</call>", r"<call>\1</call>", call)
        res = str(t0.get("result", ""))
        if res.startswith("ERROR:"):
            return None
        return (
            f"{_sys()}\n\nUser: {user}\nAssistant: {call}\n"
            f"Tool: {format_result(res)}\nAssistant: {final}\n"
        )

    # Name-only tools list (battery summary): rebuild via host_assist_call
    if names:
        assisted = host_assist_call(user)
        if not assisted:
            return None
        parsed = parse_call(assisted)
        if parsed is None:
            return None
        name, args, kwargs = parsed
        if name.lower() not in names and names[0] != name.lower():
            # still allow if assist matches first tool intent loosely
            if name.lower() != names[0]:
                return None
        res = execute_call(name, list(args), dict(kwargs))
        if res.startswith("ERROR:"):
            return None
        return (
            f"{_sys()}\n\nUser: {user}\nAssistant: {assisted}\n"
            f"Tool: {format_result(res)}\nAssistant: {final}\n"
        )

    # no-call win
    if "<call>" in final.lower():
        return None
    return f"{_sys()}\n\nUser: {user}\nAssistant: {final}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battery", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, default="data/tool_sft_self_distill.txt")
    ap.add_argument("--min-pass-rate", type=float, default=0.0)
    args = ap.parse_args()

    episodes: list[str] = []
    stats: dict[str, int] = {}
    for path_s in args.battery:
        path = Path(path_s)
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        pr = float(d.get("pass_rate") or 0.0)
        if pr < args.min_pass_rate:
            print(f"skip {path} pass_rate={pr:.3f} < {args.min_pass_rate}")
            continue
        for case in d.get("all") or []:
            if not isinstance(case, dict):
                continue
            ep = _episode_from_case(case)
            if ep is None:
                continue
            cat = case.get("category") or "other"
            stats[cat] = stats.get(cat, 0) + 1
            episodes.append(ep)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    uniq = list(dict.fromkeys(episodes))
    out.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")
    print(f"wrote {out}  episodes={len(uniq)} (from {len(episodes)} raw)  by_cat={stats}")


if __name__ == "__main__":
    main()
