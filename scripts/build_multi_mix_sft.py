#!/usr/bin/env python3
"""Build multi-objective SFT text: dense replay + fact Q&A + tool episodes.

Prevents tool-only SFT from wiping closed-book / language signal (Exp D SciQ drop).

Default char mix (approx):
  dense 50% | fact 20% | tool 30%

  uv run python scripts/build_multi_mix_sft.py \\
    --dense data/base_en_dense/mix_train.txt \\
    --out data/multi_mix_sft.txt
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path


def _sample_docs(path: Path, max_chars: int, rng: random.Random) -> str:
    if max_chars <= 0 or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        return text[:max_chars]
    rng.shuffle(parts)
    out: list[str] = []
    n = 0
    for p in parts:
        if n >= max_chars:
            break
        out.append(p)
        n += len(p) + 2
    return "\n\n".join(out)


def _run(cmd: list[str], cwd: Path) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense", type=str, default="data/base_en_dense/mix_train.txt")
    ap.add_argument("--out", type=str, default="data/multi_mix_sft.txt")
    ap.add_argument("--val-out", type=str, default="data/multi_mix_sft_val.txt")
    ap.add_argument("--total-chars", type=int, default=40_000_000)
    ap.add_argument("--dense-frac", type=float, default=0.50)
    ap.add_argument("--fact-frac", type=float, default=0.20)
    ap.add_argument("--tool-frac", type=float, default=0.30)
    ap.add_argument("--fact-n", type=int, default=20_000)
    ap.add_argument("--tool-n", type=int, default=8_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kb", type=str, default="ll3m/kb_facts.json")
    ap.add_argument(
        "--tool-extra",
        type=str,
        default="",
        help="optional extra tool SFT text to append (e.g. self-distill harvest)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    rng = random.Random(args.seed)
    py = sys.executable

    dense_budget = int(args.total_chars * args.dense_frac)
    fact_budget = int(args.total_chars * args.fact_frac)
    tool_budget = int(args.total_chars * args.tool_frac)

    fact_path = root / "data" / "fact_qa_sft_mix.txt"
    tool_path = root / "data" / "tool_sft_mix.txt"
    fact_path.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            py,
            str(root / "scripts" / "gen_fact_qa_sft.py"),
            "--out",
            str(fact_path),
            "--n",
            str(args.fact_n),
            "--seed",
            str(args.seed),
            "--kb",
            str(root / args.kb),
        ],
        root,
    )
    _run(
        [
            py,
            str(root / "scripts" / "gen_tool_sft.py"),
            "--out",
            str(tool_path),
            "--n",
            str(args.tool_n),
            "--seed",
            str(args.seed + 1),
        ],
        root,
    )

    dense = _sample_docs(Path(args.dense), dense_budget, rng)
    fact = fact_path.read_text(encoding="utf-8")
    tool = tool_path.read_text(encoding="utf-8")
    if args.tool_extra:
        extra_p = Path(args.tool_extra)
        if extra_p.is_file():
            tool = tool + "\n\n" + extra_p.read_text(encoding="utf-8")
            print(f"appended tool-extra {extra_p} (+{extra_p.stat().st_size} bytes)", flush=True)
    if len(fact) > fact_budget:
        fact = fact[:fact_budget]
    if len(tool) > tool_budget:
        tool = tool[:tool_budget]

    # Interleave blocks so the model sees all three throughout training.
    blocks: list[str] = []
    for src, name in ((dense, "dense"), (fact, "fact"), (tool, "tool")):
        parts = [p.strip() for p in src.split("\n\n") if p.strip()]
        rng.shuffle(parts)
        blocks.extend(parts)
        print(f"{name}: {len(parts):,} docs  chars={sum(len(p) for p in parts):,}", flush=True)
    rng.shuffle(blocks)

    n_val = max(50, int(len(blocks) * 0.02))
    val_blocks = blocks[:n_val]
    train_blocks = blocks[n_val:]
    train_text = "\n\n".join(train_blocks) + "\n"
    val_text = "\n\n".join(val_blocks) + "\n"

    out = Path(args.out)
    val_out = Path(args.val_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(train_text, encoding="utf-8")
    val_out.write_text(val_text, encoding="utf-8")
    print(
        f"wrote {out} chars={len(train_text):,} docs={len(train_blocks):,}\n"
        f"wrote {val_out} chars={len(val_text):,} docs={len(val_blocks):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
