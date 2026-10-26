#!/usr/bin/env python3
"""Export a ll3m .pt checkpoint for bare-metal C inference.

Formats (4-byte magic):
  L3T2  — ternary Linears (2-bit) + int8 embeds (product default)  → *.ll3m
  L3B2  — binary  Linears (1-bit) + int8 embeds (research denser) → *.ll3m
  L3M2  — f32 weights (debug / goldens)                            → *.ll3m

Legacy aliases still readable by C: CLT2/CLM2 and *.clm.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from ll3m.config import GPTConfig
from ll3m.qat import (
    pack_binary_1bit,
    pack_ternary_2bit,
    quantize_binary_numpy,
    quantize_embed_int8,
    quantize_ternary_numpy,
)

# Product file extension
WEIGHTS_EXT = ".ll3m"
MAGIC_TERNARY = b"L3T2"
MAGIC_BINARY = b"L3B2"
MAGIC_F32 = b"L3M2"


def _flags(has_ls: bool, has_peri: bool) -> int:
    return (1 if has_ls else 0) | (2 if has_peri else 0)


def _header(
    magic: bytes,
    cfg: GPTConfig,
    flags: int,
    tok_len: int,
) -> bytes:
    V, T, L = cfg.vocab_size, cfg.block_size, cfg.n_layer
    H, nkv, D = cfg.n_head, cfg.kv_head, cfg.n_embd
    mid = cfg.intermediate
    theta = float(cfg.rope_theta)
    return struct.pack(
        "<4s7IfII",
        magic,
        V,
        T,
        L,
        H,
        nkv,
        D,
        mid,
        theta,
        flags,
        tok_len,
    )


def _pack_v2_f32(ck: dict, cfg: GPTConfig, tok_bytes: bytes) -> bytes:
    """L3M2: f32 weights in map order for c_infer v2."""
    if "params" not in ck:
        raise KeyError("checkpoint missing 'params' dict")
    p = ck["params"]
    L = cfg.n_layer
    blobs: list[np.ndarray] = []

    def add(name: str) -> None:
        if name not in p:
            raise KeyError(f"checkpoint missing weight {name!r}")
        t = p[name].detach().contiguous().float().cpu().numpy().reshape(-1)
        blobs.append(t)

    add("transformer.wte.weight")
    has_ls = any(k.endswith(".ls_1") for k in p)
    has_peri = any(".post_attn.weight" in k for k in p)
    for i in range(L):
        pref = f"transformer.h.{i}."
        add(pref + "rms_1.weight")
        add(pref + "attn.q_proj.weight")
        add(pref + "attn.k_proj.weight")
        add(pref + "attn.v_proj.weight")
        add(pref + "attn.o_proj.weight")
        add(pref + "attn.q_norm.weight")
        add(pref + "attn.k_norm.weight")
        add(pref + "rms_2.weight")
        add(pref + "mlp.gate.weight")
        add(pref + "mlp.up.weight")
        add(pref + "mlp.down.weight")
        if has_ls:
            add(pref + "ls_1")
            add(pref + "ls_2")
        if has_peri:
            add(pref + "post_attn.weight")
            add(pref + "post_mlp.weight")
    add("transformer.ln_f.weight")
    flat = np.concatenate(blobs).astype(np.float32)
    return (
        _header(MAGIC_F32, cfg, _flags(has_ls, has_peri), len(tok_bytes))
        + tok_bytes
        + flat.tobytes()
    )


def _pack_linear_ternary(w: np.ndarray) -> bytes:
    """Ternary pack one (out, in) Linear → scale[out] f32 + packed uint8."""
    q, sc = quantize_ternary_numpy(w)
    packed = pack_ternary_2bit(q)
    return sc.astype(np.float32).tobytes() + packed.tobytes()


def _pack_linear_binary(w: np.ndarray) -> bytes:
    """Binary pack one (out, in) Linear → scale[out] f32 + 1-bit packed uint8."""
    q, sc = quantize_binary_numpy(w)
    packed = pack_binary_1bit(q)
    return sc.astype(np.float32).tobytes() + packed.tobytes()


def _pack_v2_quant_linears(
    ck: dict,
    cfg: GPTConfig,
    tok_bytes: bytes,
    *,
    magic: bytes,
    pack_linear,
    flag_bit: int,
) -> bytes:
    """Shared L3T2/L3B2 layout: f32 norms + int8 embeds + packed Linears."""
    if "params" not in ck:
        raise KeyError("checkpoint missing 'params' dict")
    p = ck["params"]
    L = cfg.n_layer
    has_ls = any(k.endswith(".ls_1") for k in p)
    has_peri = any(".post_attn.weight" in k for k in p)

    def t(name: str) -> np.ndarray:
        if name not in p:
            raise KeyError(f"checkpoint missing weight {name!r}")
        return p[name].detach().contiguous().float().cpu().numpy()

    parts: list[bytes] = []
    for i in range(L):
        pref = f"transformer.h.{i}."
        parts.append(t(pref + "rms_1.weight").astype(np.float32).tobytes())
        parts.append(t(pref + "attn.q_norm.weight").astype(np.float32).tobytes())
        parts.append(t(pref + "attn.k_norm.weight").astype(np.float32).tobytes())
        parts.append(t(pref + "rms_2.weight").astype(np.float32).tobytes())
        if has_ls:
            parts.append(t(pref + "ls_1").astype(np.float32).tobytes())
            parts.append(t(pref + "ls_2").astype(np.float32).tobytes())
        if has_peri:
            parts.append(t(pref + "post_attn.weight").astype(np.float32).tobytes())
            parts.append(t(pref + "post_mlp.weight").astype(np.float32).tobytes())
    parts.append(t("transformer.ln_f.weight").astype(np.float32).tobytes())

    wte = t("transformer.wte.weight")
    eq, esc = quantize_embed_int8(wte)
    parts.append(eq.tobytes())
    parts.append(esc.astype(np.float32).tobytes())

    for i in range(L):
        pref = f"transformer.h.{i}."
        for name in (
            "attn.q_proj.weight",
            "attn.k_proj.weight",
            "attn.v_proj.weight",
            "attn.o_proj.weight",
            "mlp.gate.weight",
            "mlp.up.weight",
            "mlp.down.weight",
        ):
            parts.append(pack_linear(t(pref + name)))

    flags = _flags(has_ls, has_peri) | flag_bit
    return (
        _header(magic, cfg, flags, len(tok_bytes))
        + tok_bytes
        + b"".join(parts)
    )


def _pack_v2_ternary(ck: dict, cfg: GPTConfig, tok_bytes: bytes) -> bytes:
    """L3T2: f32 norms + int8 embeds + ternary Linears (offline pack)."""
    return _pack_v2_quant_linears(
        ck, cfg, tok_bytes,
        magic=MAGIC_TERNARY,
        pack_linear=_pack_linear_ternary,
        flag_bit=4,  # bit2 = ternary pack
    )


def _pack_v2_binary(ck: dict, cfg: GPTConfig, tok_bytes: bytes) -> bytes:
    """L3B2: f32 norms + int8 embeds + binary Linears (research denser pack)."""
    return _pack_v2_quant_linears(
        ck, cfg, tok_bytes,
        magic=MAGIC_BINARY,
        pack_linear=_pack_linear_binary,
        flag_bit=8,  # bit3 = binary pack
    )


def pack_checkpoint(
    path: Path,
    tokenizer_path: Path | None = None,
    *,
    quant: str = "ternary",
) -> bytes:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(ck)}")
    if "config" not in ck:
        raise KeyError("checkpoint missing 'config'")
    cfg = GPTConfig.from_dict(dict(ck["config"]))
    tok_bytes = b""
    tp = tokenizer_path
    if tp is None and ck.get("tokenizer_path"):
        tp = Path(ck["tokenizer_path"])
    if tp is None:
        cand = Path(__file__).resolve().parent.parent / "data" / "tokenizer.json"
        if cand.is_file():
            tp = cand
    if tp is not None and Path(tp).is_file():
        tok_bytes = Path(tp).read_bytes()
    if quant == "f32":
        return _pack_v2_f32(ck, cfg, tok_bytes)
    if quant == "ternary":
        return _pack_v2_ternary(ck, cfg, tok_bytes)
    if quant == "binary":
        return _pack_v2_binary(ck, cfg, tok_bytes)
    raise ValueError(f"unknown quant={quant!r} (want f32|ternary|binary)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ll3m weights for C inference")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    ap.add_argument("--out", type=str, default="checkpoints/best.ll3m")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument(
        "--quant",
        type=str,
        default="ternary",
        choices=["ternary", "binary", "f32"],
        help="ternary=L3T2 product; binary=L3B2 research 1-bit; f32=L3M2 debug",
    )
    args = ap.parse_args()
    src = Path(args.checkpoint)
    if not src.is_file():
        print(f"missing {src}", file=sys.stderr)
        sys.exit(1)
    data = pack_checkpoint(
        src,
        Path(args.tokenizer) if args.tokenizer else None,
        quant=args.quant,
    )
    out = Path(args.out)
    if out.suffix == ".clm":
        print("note: prefer .ll3m extension (legacy .clm still loadable)", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    kind = {
        "ternary": "L3T2 ternary",
        "binary": "L3B2 binary",
        "f32": "L3M2 f32",
    }.get(args.quant, args.quant)
    print(f"wrote {out} ({len(data)} bytes, {len(data)/1024/1024:.2f} MiB, {kind})")


if __name__ == "__main__":
    main()
