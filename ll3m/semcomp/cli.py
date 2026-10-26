"""CLI: estimate | run | audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ll3m.semcomp.pipeline import audit_output, estimate_only, run_pipeline, write_run_output
from ll3m.semcomp.render import RenderProfile


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ll3m.semcomp",
        description="Semantic compressor (hard meaning model M*). See docs/SEMANTIC_COMPRESSOR.md",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="extract + ratio report only (no write)")
    e.add_argument("--in", dest="in_path", required=True, help="input .txt or directory")
    e.add_argument("--tokenizer", default=None, help="optional tokenizer.json for token counts")

    r = sub.add_parser("run", help="compress and write corpus + claims + coverage")
    r.add_argument("--in", dest="in_path", required=True)
    r.add_argument("--out", dest="out_dir", required=True)
    r.add_argument(
        "--profile",
        default="atomic",
        choices=["atomic", "qa", "compact_prose", "mixed"],
    )
    r.add_argument("--meaning-model", default="hard_v1")
    r.add_argument("--tokenizer", default=None)
    r.add_argument(
        "--verify",
        default="strict",
        choices=["strict", "off"],
        help="strict: round-trip re-extract (default)",
    )
    r.add_argument(
        "--refuse-on-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="do not write mix.txt if coverage fails (default: true)",
    )

    a = sub.add_parser("audit", help="re-extract mix.txt and check coverage vs source")
    a.add_argument("--corpus", required=True, help="output dir with mix.txt")
    a.add_argument("--against", required=True, help="original input path")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "estimate":
        est = estimate_only(args.in_path, tokenizer_path=args.tokenizer)
        print(json.dumps(est, indent=2))
        return 0

    if args.cmd == "run":
        result = run_pipeline(
            args.in_path,
            profile=args.profile,  # type: ignore[arg-type]
            meaning_model=args.meaning_model,
            tokenizer_path=args.tokenizer,
            roundtrip=(args.verify == "strict"),
        )
        out = write_run_output(
            result,
            args.out_dir,
            refuse_on_residual=args.refuse_on_residual,
        )
        rep = result["report"]
        print(
            json.dumps(
                {
                    "out": str(out),
                    "ok": rep.ok,
                    "n_claims": len(result["claims"]),
                    "char_ratio": (rep.char_out / rep.char_in) if rep.char_in else None,
                    "token_ratio": (rep.token_out / rep.token_in)
                    if rep.token_in and rep.token_out
                    else None,
                    "n_missing": rep.n_missing,
                    "n_invented": rep.n_invented,
                    "notes": rep.notes,
                },
                indent=2,
            )
        )
        if args.refuse_on_residual and not rep.ok:
            return 2
        return 0

    if args.cmd == "audit":
        rep = audit_output(args.corpus, args.against)
        print(json.dumps(rep.to_dict(), indent=2))
        return 0 if rep.ok else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
