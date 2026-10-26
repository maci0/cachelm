#!/usr/bin/env python3
"""L3 CCD capacity search: max n_params under infer WS gate.

Reinvest denser Linear packing into shape (params), not free RAM.

Grids:
  - product pack:  ternary Linears + int8 embeds  (ship-ready estimate)
  - denser pack:   binary  Linears + int8 embeds  (research; no C ship yet)

Default T=256, V=2048, peri_ln=False, layer_scale_init=1e-4, MQA preferred.

  uv run python scripts/search_l3_capacity.py
  uv run python scripts/search_l3_capacity.py --out checkpoints/l3_capacity_search.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ll3m.cache_info import probe_cache
from ll3m.config import GPTConfig
from ll3m.model_torch import estimate_infer_ws


@dataclass(frozen=True)
class Cand:
    n_layer: int
    n_embd: int
    n_head: int
    n_kv_head: int
    intermediate_size: int
    block_size: int
    quant: str
    embed_quant: str
    n_params: int
    ws_total: int
    ws_weights: int
    ws_acts: int
    fill: float  # total / gate

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def _mid_options(d: int) -> list[int]:
    """64-aligned intermediate sizes around default ~8/3 d."""
    base = max(64, (int(round(d * 8 / 3)) + 63) // 64 * 64)
    opts = {base}
    for m in (2.0, 2.5, 8 / 3, 3.0, 3.5, 4.0):
        raw = int(round(d * m))
        opts.add(max(64, (raw + 63) // 64 * 64))
    return sorted(opts)


def search(
    *,
    gate: int,
    block_size: int,
    vocab_size: int,
    quant: str,
    embed_quant: str,
    max_layer: int,
    embds: list[int],
    heads: list[int],
    head_dims: list[int],
    kv_heads: list[int],
    top_k: int,
) -> list[Cand]:
    cands: list[Cand] = []
    for n_layer in range(12, max_layer + 1):
        for n_embd in embds:
            for n_head in heads:
                if n_embd % n_head != 0:
                    continue
                hd = n_embd // n_head
                if hd not in head_dims:
                    continue
                for n_kv in kv_heads:
                    if n_head % n_kv != 0:
                        continue
                    if n_embd % n_kv != 0:
                        continue
                    for mid in _mid_options(n_embd):
                        cfg = GPTConfig(
                            vocab_size=vocab_size,
                            block_size=block_size,
                            n_layer=n_layer,
                            n_head=n_head,
                            n_embd=n_embd,
                            n_kv_head=n_kv,
                            intermediate_size=mid,
                            peri_ln=False,
                            layer_scale_init=1e-4,
                        )
                        inf = estimate_infer_ws(
                            cfg, quant=quant, embed_quant=embed_quant
                        )
                        if inf.total > gate:
                            continue
                        cands.append(
                            Cand(
                                n_layer=n_layer,
                                n_embd=n_embd,
                                n_head=n_head,
                                n_kv_head=n_kv,
                                intermediate_size=mid,
                                block_size=block_size,
                                quant=quant,
                                embed_quant=embed_quant,
                                n_params=cfg.n_params(),
                                ws_total=inf.total,
                                ws_weights=inf.weight_bytes,
                                ws_acts=inf.activation_bytes,
                                fill=inf.total / gate,
                            )
                        )
    cands.sort(key=lambda c: (c.n_params, c.fill), reverse=True)
    # de-dupe near-identical shapes keep max params only per (L,d,H,kv,mid)
    return cands[:top_k]


def product_baseline(gate: int, block_size: int) -> Cand:
    cfg = GPTConfig(
        vocab_size=2048,
        block_size=block_size,
        n_layer=23,
        n_head=12,
        n_embd=576,
        n_kv_head=1,
        intermediate_size=1536,
        peri_ln=False,
        layer_scale_init=1e-4,
    )
    inf = estimate_infer_ws(cfg, quant="ternary", embed_quant="int8")
    return Cand(
        n_layer=23,
        n_embd=576,
        n_head=12,
        n_kv_head=1,
        intermediate_size=1536,
        block_size=block_size,
        quant="ternary",
        embed_quant="int8",
        n_params=cfg.n_params(),
        ws_total=inf.total,
        ws_weights=inf.weight_bytes,
        ws_acts=inf.activation_bytes,
        fill=inf.total / gate,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--gate-frac", type=float, default=0.95)
    ap.add_argument("--max-layer", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument(
        "--out",
        type=str,
        default="checkpoints/l3_capacity_search.json",
    )
    args = ap.parse_args()

    hier = probe_cache()
    gate = int(hier.l3_bytes * args.gate_frac)
    embds = list(range(384, 769, 32))  # 384..768
    heads = [8, 12, 16]
    head_dims = [32, 40, 48, 64]
    kv_heads = [1]  # MQA first (param density at T=256)

    base = product_baseline(gate, args.block_size)
    tern = search(
        gate=gate,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        quant="ternary",
        embed_quant="int8",
        max_layer=args.max_layer,
        embds=embds,
        heads=heads,
        head_dims=head_dims,
        kv_heads=kv_heads,
        top_k=args.top_k,
    )
    binary = search(
        gate=gate,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        quant="binary",
        embed_quant="int8",
        max_layer=args.max_layer,
        embds=embds,
        heads=heads,
        head_dims=head_dims,
        kv_heads=kv_heads,
        top_k=args.top_k,
    )

    def fmt(c: Cand) -> str:
        return (
            f"L{c.n_layer}×d{c.n_embd} h{c.n_head}/kv{c.n_kv_head} "
            f"mid{c.intermediate_size}  params={c.n_params:,}  "
            f"WS={c.ws_total/1024**2:.2f} MiB ({100*c.fill:.1f}% gate)  "
            f"[{c.quant}+{c.embed_quant}]"
        )

    print(f"L3 instance {hier.l3_bytes/1024**2:.1f} MiB  gate {gate/1024**2:.2f} MiB ({args.gate_frac:.0%})")
    print(f"T={args.block_size} V={args.vocab_size}  peri_ln=False  ls=1e-4  MQA search")
    print()
    print("=== product baseline (ship freeze) ===")
    print(fmt(base))
    print()
    print("=== top ternary+int8 (ship pack) ===")
    for i, c in enumerate(tern[:8], 1):
        delta = c.n_params - base.n_params
        print(f"{i:2}. {fmt(c)}  Δparams={delta:+,}")
    print()
    print("=== top binary+int8 (research reinvest; no C ship pack) ===")
    for i, c in enumerate(binary[:8], 1):
        delta = c.n_params - base.n_params
        print(f"{i:2}. {fmt(c)}  Δparams={delta:+,}")

    best_t = tern[0] if tern else None
    best_b = binary[0] if binary else None
    print()
    print("=== reinvestment summary ===")
    if best_t:
        print(
            f"ternary densest: {best_t.n_params:,} params "
            f"({100*(best_t.n_params/base.n_params - 1):+.1f}% vs freeze)  "
            f"WS {best_t.ws_total/1024**2:.2f} MiB"
        )
    if best_b:
        print(
            f"binary densest:  {best_b.n_params:,} params "
            f"({100*(best_b.n_params/base.n_params - 1):+.1f}% vs freeze)  "
            f"WS {best_b.ws_total/1024**2:.2f} MiB"
        )
        if best_t:
            print(
                f"binary vs best ternary: "
                f"{best_b.n_params - best_t.n_params:+,} params "
                f"({100*(best_b.n_params/best_t.n_params - 1):+.1f}%)"
            )
    print()
    print(
        "Note: binary pack is estimate-only until C 1-bit GEMV + QAT alphabet "
        "{-1,+1} exist. Prefer shipping a denser ternary shape first."
    )

    out = {
        "gate_bytes": gate,
        "gate_mib": gate / 1024**2,
        "l3_bytes": hier.l3_bytes,
        "block_size": args.block_size,
        "vocab_size": args.vocab_size,
        "baseline": asdict(base),
        "ternary_int8_top": [asdict(c) for c in tern],
        "binary_int8_top": [asdict(c) for c in binary],
        "recommendation": {
            "ship_pack": "ternary+int8",
            "first_reinvest": asdict(best_t) if best_t else None,
            "research_binary_reinvest": asdict(best_b) if best_b else None,
            "caveat": (
                "Binary Linear pack is capacity-search only; no product C path. "
                "Reinvest ternary denser shape under same pack before binary R&D."
            ),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
