#!/usr/bin/env python3
"""Build a tiny high-value slice from cap_mix and run semcomp.

Default: first N non-story / non-system blocks from sciq + agent mixes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ll3m.semcomp.pipeline import estimate_only, run_pipeline, write_run_output  # noqa: E402

_STORY = re.compile(r"once upon a time", re.I)
_SYS = re.compile(
    r"you are a small helper|tools\s*\(|format:\s*<call>|if no tool needed",
    re.I,
)
_KEEP = re.compile(
    r"^(?:Q:|A:|Question:|Answer:|User:|Assistant:|Tool:)",
    re.I,
)
_PROSE_FACT = re.compile(
    r"\b(is|are|was|were|means|called|causes|defined)\b",
    re.I,
)


def _blocks(text: str) -> list[str]:
    # split on blank lines
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def select_blocks(path: Path, *, max_blocks: int) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    for b in _blocks(raw):
        if _STORY.search(b):
            continue
        if _SYS.search(b) and not _KEEP.search(b):
            # pure system card
            if len(b) > 200 and not re.search(r"^User:", b, re.M):
                continue
        lines = [ln for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        # keep dialogue / QA / short fact prose
        if any(_KEEP.match(ln.strip()) for ln in lines):
            # strip embedded system boilerplate lines inside block
            cleaned = [
                ln
                for ln in lines
                if not _SYS.match(ln.strip())
                and "Tools (emit" not in ln
                and not ln.strip().startswith("calc:")
                and not ln.strip().startswith("time:")
                and not ln.strip().startswith("lookup:")
                and not ln.strip().startswith("wiki:")
                and not ln.strip().startswith("note:")
                and not ln.strip().startswith("Format:")
            ]
            if cleaned:
                out.append("\n".join(cleaned))
        elif len(b) < 400 and _PROSE_FACT.search(b) and not _STORY.search(b):
            out.append(b)
        if len(out) >= max_blocks:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sources",
        nargs="+",
        default=[
            "data/cap_mix_v16_binary_sciq.txt",
            "data/cap_mix_v15_agent.txt",
        ],
    )
    ap.add_argument("--max-blocks", type=int, default=40)
    ap.add_argument("--out-slice", type=str, default="data/semcomp_tiny/slice.txt")
    ap.add_argument("--out-run", type=str, default="data/semcomp_tiny/out")
    ap.add_argument("--profile", default="atomic", choices=["atomic", "qa", "mixed", "compact_prose"])
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    args = ap.parse_args()

    blocks: list[str] = []
    per = max(1, args.max_blocks // max(1, len(args.sources)))
    for src in args.sources:
        p = ROOT / src if not Path(src).is_absolute() else Path(src)
        if not p.exists():
            print(f"skip missing {p}", file=sys.stderr)
            continue
        got = select_blocks(p, max_blocks=per)
        for i, b in enumerate(got):
            blocks.append(f"# doc:{p.stem}_{i:03d}\n{b}")
        if len(blocks) >= args.max_blocks:
            break
    blocks = blocks[: args.max_blocks]

    slice_path = ROOT / args.out_slice
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    slice_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"wrote slice {slice_path} blocks={len(blocks)} chars={slice_path.stat().st_size}")

    tok = str(ROOT / args.tokenizer) if (ROOT / args.tokenizer).exists() else None
    est = estimate_only(slice_path, tokenizer_path=tok)
    print("estimate:", json.dumps(est, indent=2))

    result = run_pipeline(
        slice_path,
        profile=args.profile,
        tokenizer_path=tok,
        roundtrip=True,
    )
    # tiny real data: always write mix for inspection; report residual honestly
    out = write_run_output(result, ROOT / args.out_run, refuse_on_residual=False)
    rep = result["report"]
    summary = {
        "out": str(out),
        "ok": rep.ok,
        "n_claims": len(result["claims"]),
        "n_missing": rep.n_missing,
        "n_invented": rep.n_invented,
        "literal_failures": len(rep.literal_failures),
        "token_in": rep.token_in,
        "token_out": rep.token_out,
        "token_ratio": (rep.token_out / rep.token_in) if rep.token_in and rep.token_out else None,
        "char_ratio": (rep.char_out / rep.char_in) if rep.char_in else None,
        "notes": rep.notes,
        "sample_claims": [c.surface for c in result["claims"][:12]],
    }
    (out / "tiny_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if rep.n_invented == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
