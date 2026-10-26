"""Quantization-aware training + ternary/int8 pack helpers for export.

Fake-quant STE in train; real pack helpers produce C-ready buffers.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant_per_row(
    w: torch.Tensor,
    *,
    num_bits: int = 8,
    ternary: bool = False,
    binary: bool = False,
) -> torch.Tensor:
    """
    Fake-quantize weight matrix (out, in) with STE.

    int8: per-row absmax → [-127, 127]
    ternary (BitNet-style): per-row absmean scale, round to {-1, 0, 1}
    binary (research): per-row absmean scale, sign → {-1, +1} (no zero)
    """
    if w.ndim != 2:
        return w
    if ternary and binary:
        raise ValueError("fake_quant_per_row: ternary and binary are mutually exclusive")
    if ternary:
        scale = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        q = (w / scale).round().clamp(-1.0, 1.0)
        q_hard = q * scale
        return w + (q_hard - w).detach()
    if binary:
        # Pure sign: zeros become +1 (deterministic; avoids NaN from 0/0).
        scale = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        q = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        q_hard = q * scale
        return w + (q_hard - w).detach()

    qmax = float(2 ** (num_bits - 1) - 1)
    amax = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    q = (w / scale).round().clamp(-qmax, qmax)
    q_hard = q * scale
    return w + (q_hard - w).detach()


def quantize_ternary_numpy(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    w: (out, in) float32 → (q int8 in {-1,0,1}, scale float32[out])
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"quantize_ternary_numpy expects 2D, got shape {w.shape}")
    if w.size == 0:
        raise ValueError("quantize_ternary_numpy: empty weight matrix")
    scale = np.mean(np.abs(w), axis=1).astype(np.float32)
    scale = np.maximum(scale, 1e-5)
    q = np.rint(w / scale[:, None]).clip(-1, 1).astype(np.int8)
    return q, scale


def quantize_binary_numpy(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    w: (out, in) float32 → (q int8 in {-1,+1}, scale float32[out])

    Research 1-bit path (no zero). scale = per-row absmean.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"quantize_binary_numpy expects 2D, got shape {w.shape}")
    if w.size == 0:
        raise ValueError("quantize_binary_numpy: empty weight matrix")
    scale = np.mean(np.abs(w), axis=1).astype(np.float32)
    scale = np.maximum(scale, 1e-5)
    q = np.where(w >= 0, 1, -1).astype(np.int8)
    return q, scale


def pack_binary_1bit(q: np.ndarray) -> np.ndarray:
    """
    Pack int8 binary q in {-1,+1} to uint8, 8 weights per byte.

    Encoding per bit: 0 → -1, 1 → +1 (LSB = first column).
    Rows padded on the right to multiple of 8 with +1.
    Returns flat uint8 of shape (out, ceil(in/8)) raveled.
    """
    q = np.asarray(q, dtype=np.int8)
    if q.ndim != 2:
        raise ValueError(f"pack_binary_1bit expects 2D q, got shape {q.shape}")
    out, inn = q.shape
    if out < 0 or inn < 0:
        raise ValueError("pack_binary_1bit: negative dimensions")
    # Map anything non-positive to -1, positive to +1
    q = np.where(q >= 0, 1, -1).astype(np.int8)
    pad = (-inn) % 8
    if pad:
        q = np.pad(q, ((0, 0), (0, pad)), constant_values=1)
    # -1,1 → 0,1
    enc = ((q.astype(np.int16) + 1) // 2).astype(np.uint8)
    enc = enc.reshape(out, -1, 8)
    packed = (
        enc[:, :, 0]
        | (enc[:, :, 1] << 1)
        | (enc[:, :, 2] << 2)
        | (enc[:, :, 3] << 3)
        | (enc[:, :, 4] << 4)
        | (enc[:, :, 5] << 5)
        | (enc[:, :, 6] << 6)
        | (enc[:, :, 7] << 7)
    ).astype(np.uint8)
    return packed.reshape(-1)


def unpack_binary_1bit(packed: np.ndarray, out: int, inn: int) -> np.ndarray:
    """Inverse of pack_binary_1bit."""
    if out < 0 or inn < 0:
        raise ValueError("unpack_binary_1bit: out and inn must be >= 0")
    pad = (-inn) % 8
    width = (inn + pad) // 8
    need = out * width
    p_flat = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if p_flat.size < need:
        raise ValueError(
            f"unpack_binary_1bit: packed size {p_flat.size} < need {need} "
            f"for shape ({out},{inn})"
        )
    p = p_flat[:need].reshape(out, width)
    parts = []
    for k in range(8):
        bit = (p >> k) & 1
        parts.append(bit.astype(np.int8) * 2 - 1)  # 0→-1, 1→+1
    q = np.stack(parts, axis=-1).reshape(out, -1)[:, :inn]
    return q


def pack_ternary_2bit(q: np.ndarray) -> np.ndarray:
    """
    Pack int8 ternary q in {-1,0,1} to uint8, 4 weights per byte.

    Encoding per 2 bits: 0 → -1, 1 → 0, 2 → +1
    Rows padded on the right to multiple of 4.
    Returns flat uint8 array of shape (out, ceil(in/4)) raveled.
    Values outside {-1,0,1} are clamped so corrupt inputs cannot produce
    invalid 2-bit codes (which would decode as garbage in C).
    """
    q = np.asarray(q, dtype=np.int8)
    if q.ndim != 2:
        raise ValueError(f"pack_ternary_2bit expects 2D q, got shape {q.shape}")
    out, inn = q.shape
    if out < 0 or inn < 0:
        raise ValueError("pack_ternary_2bit: negative dimensions")
    # Clamp to legal ternary alphabet before packing
    q = np.clip(q, -1, 1).astype(np.int8, copy=False)
    pad = (-inn) % 4
    if pad:
        q = np.pad(q, ((0, 0), (0, pad)), constant_values=0)
    # map -1,0,1 → 0,1,2
    enc = (q.astype(np.int16) + 1).astype(np.uint8)
    enc = enc.reshape(out, -1, 4)
    packed = (
        enc[:, :, 0]
        | (enc[:, :, 1] << 2)
        | (enc[:, :, 2] << 4)
        | (enc[:, :, 3] << 6)
    ).astype(np.uint8)
    return packed.reshape(-1)


def quantize_embed_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Embedding / tied LM head (V, d) → int8 codes + per-row absmax scale.

    denser than f32 embeds: V*d bytes + V scales vs V*d*4 floats.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"quantize_embed_int8 expects 2D, got shape {w.shape}")
    if w.size == 0:
        raise ValueError("quantize_embed_int8: empty embed matrix")
    amax = np.max(np.abs(w), axis=1).astype(np.float32)
    amax = np.maximum(amax, 1e-8)
    scale = amax / 127.0
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def pack_weights_compact(
    linear_ws: list[np.ndarray],
    *,
    embed: np.ndarray | None = None,
    linear_mode: str = "ternary",
    embed_mode: str = "int8",
) -> dict[str, object]:
    """Pack Linears (+ optional embed) into densest serve buffers.

    Returns dict with packed arrays and byte counts for L3 accounting.
    """
    out: dict[str, object] = {"linears": [], "linear_bytes": 0, "embed_bytes": 0}
    for w in linear_ws:
        w = np.asarray(w, dtype=np.float32)
        if linear_mode == "ternary":
            q, sc = quantize_ternary_numpy(w)
            packed = pack_ternary_2bit(q)
            out["linears"].append({"q_packed": packed, "scale": sc, "shape": w.shape})
            out["linear_bytes"] = int(out["linear_bytes"]) + packed.nbytes + sc.nbytes
        elif linear_mode == "binary":
            q, sc = quantize_binary_numpy(w)
            packed = pack_binary_1bit(q)
            out["linears"].append({"q_packed": packed, "scale": sc, "shape": w.shape})
            out["linear_bytes"] = int(out["linear_bytes"]) + packed.nbytes + sc.nbytes
        elif linear_mode == "int8":
            amax = np.max(np.abs(w), axis=1).astype(np.float32)
            amax = np.maximum(amax, 1e-8)
            sc = amax / 127.0
            q = np.rint(w / sc[:, None]).clip(-127, 127).astype(np.int8)
            out["linears"].append({"q": q, "scale": sc, "shape": w.shape})
            out["linear_bytes"] = int(out["linear_bytes"]) + q.nbytes + sc.nbytes
        else:
            out["linears"].append({"w": w})
            out["linear_bytes"] = int(out["linear_bytes"]) + w.nbytes
    if embed is not None:
        e = np.asarray(embed, dtype=np.float32)
        if embed_mode == "int8":
            q, sc = quantize_embed_int8(e)
            out["embed"] = {"q": q, "scale": sc, "shape": e.shape}
            out["embed_bytes"] = q.nbytes + sc.nbytes
        elif embed_mode == "ternary":
            q, sc = quantize_ternary_numpy(e)
            packed = pack_ternary_2bit(q)
            out["embed"] = {"q_packed": packed, "scale": sc, "shape": e.shape}
            out["embed_bytes"] = packed.nbytes + sc.nbytes
        elif embed_mode == "binary":
            q, sc = quantize_binary_numpy(e)
            packed = pack_binary_1bit(q)
            out["embed"] = {"q_packed": packed, "scale": sc, "shape": e.shape}
            out["embed_bytes"] = packed.nbytes + sc.nbytes
        else:
            out["embed"] = {"w": e}
            out["embed_bytes"] = e.nbytes
    out["total_weight_bytes"] = int(out["linear_bytes"]) + int(out["embed_bytes"])
    return out


def unpack_ternary_2bit(packed: np.ndarray, out: int, inn: int) -> np.ndarray:
    """Inverse of pack_ternary_2bit (for tests / golden checks)."""
    if out < 0 or inn < 0:
        raise ValueError("unpack_ternary_2bit: out and inn must be >= 0")
    pad = (-inn) % 4
    width = (inn + pad) // 4
    need = out * width
    p_flat = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if p_flat.size < need:
        raise ValueError(
            f"unpack_ternary_2bit: packed size {p_flat.size} < need {need} "
            f"for shape ({out},{inn})"
        )
    p = p_flat[:need].reshape(out, width)
    parts = []
    for k in range(4):
        e = (p >> (2 * k)) & 3
        # Codes 0,1,2 are legal; 3 is invalid 2-bit — map to 0 (neutral)
        e = np.where(e == 3, 1, e)
        parts.append(e.astype(np.int8) - 1)
    q = np.stack(parts, axis=-1).reshape(out, -1)[:, :inn]
    return q


def enable_qat_on_model(
    model: nn.Module,
    *,
    num_bits: int = 8,
    ternary: bool = False,
    binary: bool = False,
    skip_names: tuple[str, ...] = ("wte", "embed", "wpe"),
) -> int:
    """Patch nn.Linear forwards to fake-quant weights (STE)."""
    if ternary and binary:
        raise ValueError("enable_qat_on_model: ternary and binary are mutually exclusive")
    n = 0

    def _should_skip(name: str) -> bool:
        low = name.lower()
        return any(s in low for s in skip_names)

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if _should_skip(name):
            continue
        lin = mod

        def make_fwd(linear: nn.Linear, nb: int, tern: bool, binr: bool):
            def fwd(
                x: torch.Tensor,
                _linear=linear,
                _nb=nb,
                _tern=tern,
                _bin=binr,
            ) -> torch.Tensor:
                w = fake_quant_per_row(
                    _linear.weight, num_bits=_nb, ternary=_tern, binary=_bin
                )
                return F.linear(x, w, _linear.bias)

            return fwd

        lin.forward = make_fwd(lin, num_bits, ternary, binary)  # type: ignore[method-assign]
        n += 1
    return n


def enable_embed_qat_on_model(
    model: nn.Module,
    *,
    mode: str = "ternary",
) -> None:
    """Fake-quant tied wte for embedding lookup + LM head (STE).

    mode: "ternary" | "int8" | "none"
    Sets model._ll3m_embed_qat so GPT.forward uses the same grid for both.
    """
    m = mode.lower() if mode else "none"
    if m not in ("none", "int8", "ternary"):
        raise ValueError(f"unknown embed qat mode: {mode}")
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw._ll3m_embed_qat = m  # type: ignore[attr-defined]


def quantize_wte_weight(w: torch.Tensor, mode: str) -> torch.Tensor:
    """STE fake-quant for embedding / tied head weight (V, d)."""
    m = (mode or "none").lower()
    if m == "none":
        return w
    if m == "ternary":
        return fake_quant_per_row(w, ternary=True)
    if m == "int8":
        return fake_quant_per_row(w, num_bits=8, ternary=False)
    raise ValueError(f"unknown embed qat mode: {mode}")


QuantMode = Literal["none", "int8", "ternary", "binary"]
