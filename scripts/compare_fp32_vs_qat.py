#!/usr/bin/env python3
"""Compare phase-1 fp32 checkpoint vs ternary QAT checkpoint.

Reports:
  - checkpoint meta (step, qat tags, saved val_loss)
  - val CE under the *serve* grid (ternary Linears + int8 embeds via STE)
  - val CE under native ckpt mode (fp32 baseline vs QAT STE as stored)
  - fixed greedy samples

  uv run python scripts/compare_fp32_vs_qat.py \\
    --baseline checkpoints/v75/best.pt \\
    --qat checkpoints/v75_ternary/best.pt \\
    --out checkpoints/v75_ternary/quality_compare.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ll3m.qat import enable_embed_qat_on_model, enable_qat_on_model
from ll3m.train import load_model

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = [
    "Once upon a time",
    "Once upon a time, there was a person named Charles Darwin. ",
    "The little girl",
]


def _val_path() -> Path:
    p = ROOT / "data" / "mix_val.bin"
    if p.is_file():
        return p
    return ROOT / "data" / "mix_train.bin"


@torch.no_grad()
def val_ce(
    model,
    *,
    n_batches: int = 40,
    B: int = 8,
    seed: int = 0,
) -> float:
    cfg = model.config
    val = np.fromfile(_val_path(), dtype=np.uint16)
    T = min(cfg.block_size, 256)
    if len(val) < T + 2:
        raise RuntimeError("val corpus too short")
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(n_batches):
        starts = rng.integers(0, len(val) - T - 1, size=B)
        x = torch.stack(
            [torch.from_numpy(val[s : s + T].astype(np.int64)) for s in starts]
        )
        y = torch.stack(
            [torch.from_numpy(val[s + 1 : s + T + 1].astype(np.int64)) for s in starts]
        )
        logits, _ = model(x, y)
        losses.append(
            F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        )
    return float(sum(losses) / len(losses))


@torch.no_grad()
def greedy_sample(model, bpe, prompt: str, n: int = 64) -> str:
    ids = torch.tensor([bpe.encode(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=n, temperature=0.01, top_k=1)
    text = bpe.decode(out[0].tolist())
    return text.replace("\n", "\\n")


def eval_checkpoint(
    path: Path,
    *,
    force_serve_grid: bool,
    n_batches: int,
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model, _stoi, _itos, bpe = load_model(path, device="cpu")
    model.eval()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if bpe is None:
        raise RuntimeError(f"checkpoint {path} has no BPE tokenizer; cannot sample")
    if force_serve_grid:
        # Product pack grid regardless of how it was trained
        enable_qat_on_model(model, ternary=True)
        enable_embed_qat_on_model(model, mode="int8")
        grid = "ternary_lin+int8_emb (forced)"
    else:
        grid = f"native qat={ck.get('qat')} embed_qat={ck.get('embed_qat', 'none')}"

    nll = val_ce(model, n_batches=n_batches)
    samples = {
        p: greedy_sample(model, bpe, p)[:220] for p in PROMPTS
    }
    return {
        "path": str(path),
        "step": ck.get("step"),
        "ckpt_val_loss": ck.get("val_loss"),
        "qat": ck.get("qat", "none"),
        "embed_qat": ck.get("embed_qat", "none"),
        "eval_grid": grid,
        "val_ce": nll,
        "val_ppl": math.exp(min(nll, 20)),
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, required=True, help="fp32 phase-1 best.pt")
    ap.add_argument("--qat", type=str, required=True, help="ternary QAT best.pt")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    base = Path(args.baseline)
    qat = Path(args.qat)
    report: dict = {"baseline": {}, "qat": {}, "delta": {}}

    print("=== baseline (fp32 train) under serve grid ===")
    report["baseline"]["serve_grid"] = eval_checkpoint(
        base, force_serve_grid=True, n_batches=args.batches
    )
    print(
        f"  val_ce={report['baseline']['serve_grid']['val_ce']:.4f}  "
        f"ppl={report['baseline']['serve_grid']['val_ppl']:.2f}"
    )

    print("=== baseline native fp32 (no fake-quant) ===")
    report["baseline"]["native"] = eval_checkpoint(
        base, force_serve_grid=False, n_batches=args.batches
    )
    print(
        f"  val_ce={report['baseline']['native']['val_ce']:.4f}  "
        f"ppl={report['baseline']['native']['val_ppl']:.2f}"
    )

    print("=== QAT checkpoint under serve grid ===")
    report["qat"]["serve_grid"] = eval_checkpoint(
        qat, force_serve_grid=True, n_batches=args.batches
    )
    print(
        f"  val_ce={report['qat']['serve_grid']['val_ce']:.4f}  "
        f"ppl={report['qat']['serve_grid']['val_ppl']:.2f}"
    )

    print("=== QAT native (should already be ternary STE) ===")
    report["qat"]["native"] = eval_checkpoint(
        qat, force_serve_grid=False, n_batches=args.batches
    )
    print(
        f"  val_ce={report['qat']['native']['val_ce']:.4f}  "
        f"ppl={report['qat']['native']['val_ppl']:.2f}"
    )

    b = report["baseline"]["serve_grid"]["val_ce"]
    q = report["qat"]["serve_grid"]["val_ce"]
    report["delta"] = {
        "serve_grid_val_ce": q - b,
        "serve_grid_better": q < b,
        "note": "negative delta = QAT lower CE (better) under ternary+int8 serve grid",
    }
    print(
        f"\n=== delta (QAT - baseline) on serve grid: {q - b:+.4f} "
        f"({'QAT better' if q < b else 'baseline better'}) ==="
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
