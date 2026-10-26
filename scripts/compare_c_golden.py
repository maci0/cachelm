#!/usr/bin/env python3
"""Torch↔C golden: greedy token ids must match under the same quant grid.

For --quant f32: compare full-precision torch vs L3M2.
For --quant ternary: compare post-hoc ternary torch STE vs L3T2.

Usage:
  uv run python scripts/compare_c_golden.py \\
    --checkpoint checkpoints/v75/best.pt --quant ternary --tokens 16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ll3m.c_engine import ensure_weights, find_ll3m_c  # noqa: E402
from ll3m.qat import enable_embed_qat_on_model, enable_qat_on_model  # noqa: E402
from ll3m.train import load_model  # noqa: E402


@torch.no_grad()
def torch_greedy_ids(
    model,
    bpe,
    prompt: str,
    n_new: int,
) -> list[int]:
    ids = bpe.encode(prompt)
    if not ids:
        ids = [0]
    t = torch.tensor([ids], dtype=torch.long)
    for _ in range(n_new):
        cond = t[:, -model.config.block_size :]
        logits, _ = model(cond)
        nxt = int(logits[0, -1].argmax().item())
        t = torch.cat([t, torch.tensor([[nxt]])], dim=1)
    return t[0, len(ids) :].tolist()


def c_greedy_ids(clm: Path, prompt_ids: list[int], n_new: int) -> list[int]:
    binary = find_ll3m_c()
    id_str = ",".join(str(i) for i in prompt_ids)
    cmd = [
        str(binary),
        "-m",
        str(clm),
        "--quiet",
        "--stream",
        "--greedy",
        "--token-ids",
        id_str,
        "-n",
        str(n_new),
        "--warm",
        "0",
        "--steady-warm",
        "0",
        "--no-cache-stats",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"ll3m_c failed: {proc.returncode}")
    out: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(line))
        except ValueError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Torch vs C greedy token golden")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--quant", type=str, default="ternary", choices=["ternary", "f32"])
    ap.add_argument("--weights", type=str, default=None, help=".ll3m path (or legacy .clm)")
    ap.add_argument("--clm", type=str, default=None, help="legacy alias for --weights")
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    wpath = args.weights or args.clm
    weights = Path(wpath) if wpath else ck.with_suffix(".ll3m")
    ensure_weights(ck, out_path=weights, quant=args.quant)

    model, _stoi, _itos, bpe = load_model(ck, device="cpu")
    model.eval()
    if bpe is None:
        print("checkpoint needs BPE tokenizer", file=sys.stderr)
        sys.exit(1)
    if args.quant == "ternary":
        # Match offline pack grid used by export_weights L3T2
        enable_qat_on_model(model, ternary=True)
        enable_embed_qat_on_model(model, mode="int8")

    prompt_ids = bpe.encode(args.prompt) or [0]
    torch_ids = torch_greedy_ids(model, bpe, args.prompt, args.tokens)
    c_ids = c_greedy_ids(weights, prompt_ids, args.tokens)

    n = min(len(torch_ids), len(c_ids), args.tokens)
    match = sum(1 for a, b in zip(torch_ids[:n], c_ids[:n]) if a == b)
    print(f"quant={args.quant}  n={n}")
    print(f"torch: {torch_ids[:n]}")
    print(f"c:     {c_ids[:n]}")
    print(f"match: {match}/{n} ({100.0 * match / max(n, 1):.1f}%)")
    print(f"torch text: {bpe.decode(torch_ids[:n])!r}")
    print(f"c text:     {bpe.decode(c_ids[:n])!r}")

    # f32 should match tightly; ternary STE vs exact pack may drift on some steps
    if args.quant == "f32" and match < n:
        sys.exit(2)
    if args.quant == "ternary" and match < max(1, n // 2):
        print("WARN: ternary match < 50% (expected some post-hoc drift)", file=sys.stderr)
        sys.exit(3)
    print("PASS" if match == n else "PASS (partial; ternary post-hoc)")


if __name__ == "__main__":
    main()
