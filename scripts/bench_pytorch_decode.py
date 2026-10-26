#!/usr/bin/env python3
"""Benchmark PyTorch AR decode with/without KV cache (CUDA or CPU).

Example (spark1):
  .venv/bin/python scripts/bench_pytorch_decode.py \\
    --checkpoint checkpoints/v76_dense_c/best.pt --device cuda
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import torch

from ll3m.config import GPTConfig
from ll3m.model_torch import GPT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--tf32", action="store_true", default=True)
    args = ap.parse_args()

    if args.device.startswith("cuda") and args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    allowed = {f.name for f in fields(GPTConfig)}
    cfg = GPTConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    m = GPT(cfg)
    m.load_state_dict(ck["params"], strict=True)
    m.to(args.device).eval()

    print(
        f"device={args.device} shape=L{cfg.n_layer}d{cfg.n_embd} "
        f"kv={cfg.n_kv_head} T={cfg.block_size}"
    )

    def run(batch: int, use_cache: bool, n_new: int, prompt_len: int) -> float:
        g = torch.Generator(device=args.device)
        g.manual_seed(0)
        ids = torch.randint(
            0, cfg.vocab_size, (batch, prompt_len), device=args.device, generator=g
        )
        with torch.inference_mode():
            m.generate(ids.clone(), max_new_tokens=8, temperature=0.0, use_cache=use_cache)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            ids = torch.randint(
                0, cfg.vocab_size, (batch, prompt_len), device=args.device, generator=g
            )
            t0 = time.perf_counter()
            m.generate(ids, max_new_tokens=n_new, temperature=0.0, use_cache=use_cache)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        return (batch * n_new) / dt

    # Correctness
    with torch.inference_mode():
        ids = torch.randint(0, cfg.vocab_size, (1, 16), device=args.device)
        a = m.generate(ids.clone(), max_new_tokens=24, temperature=0.0, use_cache=False)
        b = m.generate(ids.clone(), max_new_tokens=24, temperature=0.0, use_cache=True)
        print("greedy cache vs nocache match:", bool(torch.equal(a, b)))

    for use_cache in (False, True):
        tps = run(1, use_cache, args.n_new, args.prompt_len)
        print(f"B=1 cache={use_cache} tok/s={tps:.1f}")

    for b in [int(x) for x in args.batches.split(",") if x.strip()]:
        tps = run(b, True, args.n_new, args.prompt_len)
        print(f"B={b} cache=True total_tok/s={tps:.1f} per_stream={tps/b:.1f}")


if __name__ == "__main__":
    main()
