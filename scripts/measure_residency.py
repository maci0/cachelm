#!/usr/bin/env python3
"""Measure C hard residency (size + mlock + tok/s + DRAM% of L2 fills).

Encodes a short BPE prompt and runs ll3m_c --residency-hard with token-ids.
Writes a report under checkpoints/compare/.

Usage:
  uv run python scripts/measure_residency.py
  uv run python scripts/measure_residency.py --ccd 1 --cpu 15 -n 256
  uv run python scripts/measure_residency.py --pack checkpoints/v76_dense_c_p1/best.ll3m
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _quiet_cpu() -> int:
    try:
        out = subprocess.check_output(
            ["bash", str(ROOT / "scripts/residency_host_setup.sh"), "quiet-cpu"],
            text=True,
        ).strip()
        return int(out)
    except Exception:
        return max(0, (os_cpu_count() or 2) - 1)


def os_cpu_count() -> int | None:
    try:
        return int(subprocess.check_output(["nproc"], text=True).strip())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pack",
        type=Path,
        default=ROOT / "checkpoints/v76_dense_c_pure_v2/best.ll3m",
    )
    ap.add_argument("-n", type=int, default=256, help="new tokens")
    ap.add_argument("--ccd", type=int, default=1, help="L3 CCD index (default 1=quieter on 9950X)")
    ap.add_argument("--cpu", type=int, default=-1, help="pin CPU (-1=auto quiet)")
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report path (default under checkpoints/compare/)",
    )
    args = ap.parse_args()

    pack = args.pack if args.pack.is_absolute() else ROOT / args.pack
    if not pack.is_file():
        print(f"missing pack: {pack}", file=sys.stderr)
        return 1
    tok_path = Path(str(pack) + ".tokenizer.json")
    if not tok_path.is_file():
        tok_path = ROOT / "data" / "tokenizer.json"
    if not tok_path.is_file():
        print(f"missing tokenizer near {pack}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT))
    from ll3m.tokenizer import BPETokenizer

    ids = BPETokenizer.load(tok_path).encode(args.prompt)
    if not ids:
        ids = [0]
    id_str = ",".join(map(str, ids))

    cpu = args.cpu if args.cpu >= 0 else _quiet_cpu()
    bin_path = ROOT / "c_infer" / "ll3m_c"
    if not bin_path.is_file():
        subprocess.check_call(["make", "-C", str(ROOT / "c_infer"), "-j"])

    cmd = [
        str(bin_path),
        "-m",
        str(pack),
        "--token-ids",
        id_str,
        "-n",
        str(args.n),
        "--greedy",
        "--residency-hard",
        "--ccd",
        str(args.ccd),
        "--cpu",
        str(cpu),
    ]
    print("run:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    out = (r.stdout or "") + "\n" + (r.stderr or "")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or (
        ROOT
        / "checkpoints"
        / "compare"
        / f"residency_hard_ccd{args.ccd}_cpu{cpu}_n{args.n}_{stamp}.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")

    def grab(pat: str) -> str | None:
        m = re.search(pat, out, re.I | re.M)
        return m.group(0).strip() if m else None

    summary = {
        "exit": r.returncode,
        "pack": str(pack),
        "ccd": args.ccd,
        "cpu": cpu,
        "n": args.n,
        "tok_s": grab(r"--- \d+ new tokens in .*? \(.*? tok/s\) ---"),
        "fits": grab(r"fits one CCD L3:.*"),
        "mlock": grab(r"model pages:.*resident.*"),
        "sched": grab(r"SCHED_FIFO:.*"),
        "bind": grab(r"CPU bind:.*"),
        "local_l3": grab(r"local L3 \(this CCD\):.*"),
        "dram": grab(r"DRAM \(near\+far\):.*"),
        "verdict": grab(r"L3 residency \(fill source\):.*"),
        "report": str(out_path),
    }
    print("--- summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"wrote {out_path}")
    return 0 if r.returncode == 0 else r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
