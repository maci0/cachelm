#!/usr/bin/env python3
"""Compare embed-int8 vs embed-ternary QAT checkpoints on val NLL + samples."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ll3m.train import load_model

ROOT = Path(__file__).resolve().parent.parent


def val_ce(ckpt: Path, n_batches: int = 50, B: int = 8, seed: int = 0) -> float:
    m, _, _, bpe = load_model(ckpt, device="cpu")
    m.eval()
    cfg = m.config
    val = np.fromfile(ROOT / "data" / "mix_val.bin", dtype=np.uint16)
    T = min(cfg.block_size, 256)
    rng = np.random.default_rng(seed)
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            starts = rng.integers(0, len(val) - T - 1, size=B)
            x = torch.stack(
                [torch.from_numpy(val[s : s + T].astype(np.int64)) for s in starts]
            )
            y = torch.stack(
                [
                    torch.from_numpy(val[s + 1 : s + T + 1].astype(np.int64))
                    for s in starts
                ]
            )
            logits, _ = m(x, y)
            losses.append(
                F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            )
    return float(sum(losses) / len(losses))


def sample(ckpt: Path, n: int = 80) -> str:
    m, _, _, bpe = load_model(ckpt, device="cpu")
    m.eval()
    ids = torch.tensor([bpe.encode("Once upon a time")], dtype=torch.long)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=n, temperature=0.8, top_k=40)
    return bpe.decode(out[0]).replace("\n", " ")


def main() -> None:
    pairs = [
        ("embed_ternary", ROOT / "checkpoints/exp_embed_ternary/best.pt"),
        ("embed_int8", ROOT / "checkpoints/exp_embed_int8/best.pt"),
        ("baseline_tern_lin", ROOT / "checkpoints/v2_dense_ternary/best.pt"),
    ]
    rows = []
    for name, p in pairs:
        if not p.is_file():
            print(f"skip missing {p}")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        nll = val_ce(p)
        rows.append(
            {
                "name": name,
                "path": str(p),
                "step": ck.get("step"),
                "ckpt_val": ck.get("val_loss"),
                "qat": ck.get("qat"),
                "embed_qat": ck.get("embed_qat", "none"),
                "val_ce": nll,
                "ppl": math.exp(min(nll, 20)),
                "sample": sample(p)[:180],
            }
        )
        print(
            f"{name:20} embed_qat={rows[-1]['embed_qat']:7}  "
            f"val_ce={nll:.4f}  ppl={rows[-1]['ppl']:.2f}  "
            f"ckpt_val={rows[-1]['ckpt_val']}"
        )
        print(f"  sample: {rows[-1]['sample'][:140]}")

    out = ROOT / "checkpoints" / "embed_qat_compare.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    if len(rows) >= 2:
        by = {r["name"]: r for r in rows}
        if "embed_ternary" in by and "embed_int8" in by:
            d = by["embed_ternary"]["val_ce"] - by["embed_int8"]["val_ce"]
            print(
                f"\nternary embeds − int8 embeds: ΔCE = {d:+.4f} "
                f"({'ternary worse' if d > 0 else 'ternary better'})"
            )


if __name__ == "__main__":
    main()
