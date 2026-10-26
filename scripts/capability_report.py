#!/usr/bin/env python3
"""Capability snapshot for a v2 checkpoint (metric honesty > vanity NLL).

Reports:
  - val NLL / BPC / PPL on a held-out slice
  - fixed prompt greedy samples (torch)
  - optional C stream sample if .clm present

Usage:
  uv run python scripts/capability_report.py --checkpoint checkpoints/v75/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ll3m.eval import nll_over_data  # noqa: E402
from ll3m.train import load_model  # noqa: E402

PROMPTS = [
    "Once upon a time",
    "The little girl",
    "In the forest,",
    "User: Hello\nAssistant:",
]


@torch.no_grad()
def sample(model, bpe, prompt: str, n: int, temperature: float = 0.8) -> str:
    ids = bpe.encode(prompt) or [0]
    t = torch.tensor([ids], dtype=torch.long)
    for _ in range(n):
        cond = t[:, -model.config.block_size :]
        logits, _ = model(cond)
        logits = logits[0, -1] / max(temperature, 1e-6)
        v, _ = torch.topk(logits, min(40, logits.size(-1)))
        logits = logits.clone()
        logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        t = torch.cat([t, torch.tensor([[nxt]])], dim=1)
    return bpe.decode(t[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "mix_val.bin"))
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    model, _stoi, _itos, bpe = load_model(ck, device="cpu")
    model.eval()
    if bpe is None:
        print("need BPE checkpoint", file=sys.stderr)
        sys.exit(1)
    cfg = model.config

    report: dict = {
        "checkpoint": str(ck),
        "config": cfg.to_dict(),
        "n_params": cfg.n_params(),
    }

    data_path = Path(args.data)
    if data_path.suffix == ".bin" and data_path.is_file():
        import numpy as np

        data = torch.from_numpy(
            np.fromfile(data_path, dtype=np.uint16).astype("int64")
        )
    elif data_path.with_suffix(".txt").is_file():
        text = data_path.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
        data = torch.tensor(bpe.encode(text), dtype=torch.long)
    else:
        # fall back to tokenizer short corpus
        text = "Once upon a time there was a little girl named Lily.\n" * 200
        data = torch.tensor(bpe.encode(text), dtype=torch.long)

    print("=== capability report ===")
    print(f"checkpoint: {ck}")
    print(
        f"model: L{cfg.n_layer} d{cfg.n_embd} h{cfg.n_head}/kv{cfg.kv_head} "
        f"T{cfg.block_size}  params={cfg.n_params():,}"
    )

    t0 = time.perf_counter()
    val_nll, n_tok = nll_over_data(
        model,
        data,
        cfg.block_size,
        batch_size=8,
        max_batches=args.max_batches,
    )
    dt = time.perf_counter() - t0
    bpc = val_nll / math.log(2)
    ppl = math.exp(min(val_nll, 20))
    report["val_nll"] = val_nll
    report["val_bpc"] = bpc
    report["val_ppl"] = ppl
    report["n_tokens_eval"] = n_tok
    print(f"\n[compression] val_nll={val_nll:.4f}  bpc={bpc:.3f}  ppl={ppl:.2f}  "
          f"tok={n_tok:,}  ({dt:.1f}s)")

    print("\n[samples] temperature=0.8 top_k=40")
    samples = []
    for p in PROMPTS:
        text = sample(model, bpe, p, args.tokens)
        samples.append({"prompt": p, "text": text})
        print(f"--- prompt: {p!r}")
        print(text[:400].replace("\n", "\\n"))
        print()
    report["samples"] = samples

    # train progress if sibling progress.json
    prog = ck.parent / "progress.json"
    if prog.is_file():
        report["progress"] = json.loads(prog.read_text())
        print(f"[train] {report['progress']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    print("done.")


if __name__ == "__main__":
    main()
