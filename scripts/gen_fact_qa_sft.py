#!/usr/bin/env python3
"""Closed-book fact Q&A SFT text from kb_facts.json (+ template variety)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/fact_qa_sft.txt")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--kb",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "ll3m" / "kb_facts.json"),
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)
    raw = json.loads(Path(args.kb).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("version") == 2:
        if isinstance(raw.get("flat"), dict) and raw["flat"]:
            kb = {str(k): str(v) for k, v in raw["flat"].items()}
        else:
            kb = {}
            for ent in raw.get("entries") or []:
                val = str(ent.get("value", ""))
                eid = str(ent.get("id", ""))
                if eid:
                    kb[eid] = val
                for a in ent.get("aliases") or []:
                    kb[str(a)] = val
    else:
        skip = {"version", "note", "n_entries", "n_keys", "entries", "flat"}
        kb = {
            str(k): str(v)
            for k, v in raw.items()
            if k not in skip and not isinstance(v, (dict, list))
        }
    items = list(kb.items())
    lines: list[str] = []
    templates = [
        ("User: What is {k}?\nAssistant: {v}.\n", 1.0),
        ("User: Tell me about {k}.\nAssistant: {v}.\n", 0.5),
        ("User: Define {k}.\nAssistant: {v}.\n", 0.4),
        ("Q: {k}?\nA: {v}\n", 0.6),
        ("Fact check: {k}\nAnswer: {v}\n", 0.3),
    ]
    for _ in range(args.n):
        k, v = rng.choice(items)
        # weight templates
        t = rng.choices([x[0] for x in templates], weights=[x[1] for x in templates], k=1)[0]
        lines.append(t.format(k=k, v=v))
        # occasional no-tool chat so we don't only do facts
        if rng.random() < 0.08:
            lines.append(
                rng.choice(
                    [
                        "User: Hello\nAssistant: Hello! How can I help?\n",
                        "User: Thanks\nAssistant: You're welcome.\n",
                        "User: How are you?\nAssistant: I'm doing well. How can I help?\n",
                    ]
                )
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} episodes~{args.n} chars={len(text):,} kb={len(items)}")


if __name__ == "__main__":
    main()
