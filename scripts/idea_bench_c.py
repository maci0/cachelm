#!/usr/bin/env python3
"""C decode idea-bench + residency gate (cwen-style A/B, L3-honest).

Reports:
  - wall tok/s (decode-only estimate via n tokens)
  - weight MiB / infer WS MiB from ll3m_c banner
  - fits one CCD L3 (YES/NO)
  - gate: ternary L3T2 should fit; f32 L3M2 usually will not

Usage:
  uv run python scripts/idea_bench_c.py --weights checkpoints/v75/best.ll3m --tokens 32
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ll3m.c_engine import find_ll3m_c, read_clm_meta  # noqa: E402
from ll3m.tokenizer import BPETokenizer  # noqa: E402


def run_once(
    clm: Path,
    ids: list[int],
    n: int,
    *,
    hard: bool,
    warm: int,
) -> dict:
    binary = find_ll3m_c()
    cmd = [
        str(binary),
        "-m",
        str(clm),
        "--token-ids",
        ",".join(str(i) for i in ids),
        "-n",
        str(n),
        "--greedy",
        "--stream",
        "--warm",
        str(warm),
        "--steady-warm",
        "0",
        "--no-cache-stats",
    ]
    if hard:
        cmd.append("--residency-hard")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    dt = time.perf_counter() - t0
    if proc.returncode not in (0, 2):
        # 2 = L3 refuse under --l3-resident/--hard
        pass
    tokens = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.isdigit() or (line.startswith("-") and line[1:].isdigit()):
            try:
                tokens.append(int(line))
            except ValueError:
                pass
    stderr = proc.stderr + "\n" + proc.stdout
    # parse weight line from stderr (load) or stdout banner
    mib = None
    m = re.search(r"weights=([\d.]+)\s*MiB", stderr)
    if m:
        mib = float(m.group(1))
    ws = None
    m = re.search(r"infer WS:\s*([\d.]+)\s*MiB", stderr)
    if m:
        ws = float(m.group(1))
    fits = None
    m = re.search(r"fits one CCD L3:\s*(YES|NO)", stderr)
    if m:
        fits = m.group(1) == "YES"
    refused = proc.returncode == 2 or "REFUSE" in stderr
    return {
        "returncode": proc.returncode,
        "dt": dt,
        "n_out": len(tokens),
        "tok_s": (len(tokens) / dt) if dt > 1e-9 else 0.0,
        "weight_mib": mib,
        "ws_mib": ws,
        "fits_ccd": fits,
        "refused": refused,
        "stderr_tail": stderr[-800:],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="C idea-bench + residency gate")
    ap.add_argument("--weights", type=str, default=None, help=".ll3m pack")
    ap.add_argument("--clm", type=str, default=None, help="legacy alias for --weights")
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--tokenizer", type=str, default=str(ROOT / "data" / "tokenizer.json"))
    args = ap.parse_args()

    warg = args.weights or args.clm
    if not warg:
        print("need --weights path.ll3m", file=sys.stderr)
        sys.exit(1)
    weights = Path(warg)
    if not weights.is_file():
        print(f"missing {weights}", file=sys.stderr)
        sys.exit(1)
    meta = read_clm_meta(weights)
    bpe = BPETokenizer.load(args.tokenizer)
    ids = bpe.encode(args.prompt) or [0]
    print(f"model: {weights.name}  magic={meta.magic}  L{meta.n_layer} d{meta.n_embd}")
    print(f"prompt tokens: {len(ids)}  generate: {args.tokens}")

    soft = run_once(weights, ids, args.tokens, hard=False, warm=args.warm)
    print("\n=== soft (default) ===")
    print(
        f"  tok/s={soft['tok_s']:.2f}  n_out={soft['n_out']}  dt={soft['dt']:.3f}s  "
        f"rc={soft['returncode']}"
    )
    if soft["weight_mib"] is not None:
        print(f"  weights={soft['weight_mib']:.2f} MiB")
    if soft["ws_mib"] is not None:
        print(f"  infer_ws={soft['ws_mib']:.2f} MiB  fits_ccd={soft['fits_ccd']}")

    hard = run_once(weights, ids, args.tokens, hard=True, warm=max(args.warm, 4))
    print("\n=== residency-hard ===")
    print(
        f"  tok/s={hard['tok_s']:.2f}  n_out={hard['n_out']}  dt={hard['dt']:.3f}s  "
        f"rc={hard['returncode']}  refused={hard['refused']}"
    )
    if hard["ws_mib"] is not None:
        print(f"  infer_ws={hard['ws_mib']:.2f} MiB  fits_ccd={hard['fits_ccd']}")
    if hard["refused"]:
        print("  gate: HARD REFUSE (WS > CCD) — expected for f32 L3M2; not for L3T2 product")
    elif hard["fits_ccd"] is True:
        print("  gate: PASS (fits one CCD L3 under hard)")
    elif meta.magic in ("L3T2", "CLT2") and hard["returncode"] == 0:
        print("  gate: ran hard without refuse (check banner for fits_ccd)")
    else:
        print("  gate: see banner / stderr")

    if meta.magic in ("L3T2", "CLT2") and hard["refused"]:
        print("\nFAIL: product L3T2 refused under residency-hard", file=sys.stderr)
        sys.exit(2)
    if meta.magic in ("L3M2", "CLM2") and not hard["refused"] and hard.get("ws_mib", 0) and hard["ws_mib"] > 40:
        print("\nNOTE: f32 L3M2 is not the L3 product pack; use --quant ternary")
    print("\ndone.")


if __name__ == "__main__":
    main()
