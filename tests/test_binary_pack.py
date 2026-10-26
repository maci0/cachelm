"""Binary 1-bit pack / QAT / L3B2 export (research denser path)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

from ll3m.config import GPTConfig
from ll3m.model_torch import GPT, estimate_infer_ws
from ll3m.qat import (
    enable_qat_on_model,
    fake_quant_per_row,
    pack_binary_1bit,
    pack_weights_compact,
    quantize_binary_numpy,
    unpack_binary_1bit,
)


def test_binary_pack_roundtrip() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((17, 33)).astype(np.float32)
    q, sc = quantize_binary_numpy(w)
    assert set(np.unique(q)).issubset({-1, 1})
    assert sc.shape == (17,)
    packed = pack_binary_1bit(q)
    back = unpack_binary_1bit(packed, 17, 33)
    assert (back == q).all()


def test_binary_pack_smaller_than_ternary() -> None:
    from ll3m.qat import pack_ternary_2bit, quantize_ternary_numpy

    rng = np.random.default_rng(1)
    w = rng.standard_normal((64, 128)).astype(np.float32)
    qt, _ = quantize_ternary_numpy(w)
    qb, _ = quantize_binary_numpy(w)
    pt = pack_ternary_2bit(qt)
    pb = pack_binary_1bit(qb)
    assert pb.nbytes < pt.nbytes
    assert pb.nbytes == 64 * ((128 + 7) // 8)


def test_fake_quant_binary_ste() -> None:
    w = torch.randn(8, 16, requires_grad=True)
    y = fake_quant_per_row(w, binary=True)
    assert y.shape == w.shape
    # reconstructed should be ±scale only
    with torch.no_grad():
        scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        ratio = (y / scale).detach()
        ok = ((ratio - 1).abs() < 1e-4) | ((ratio + 1).abs() < 1e-4)
        assert bool(ok.all())
    loss = y.sum()
    loss.backward()
    assert w.grad is not None


def test_enable_qat_binary() -> None:
    cfg = GPTConfig(64, 32, 2, 4, 64, n_kv_head=1)
    m = GPT(cfg)
    n = enable_qat_on_model(m, binary=True)
    assert n > 0
    x = torch.randint(0, 64, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape[-1] == 64
    assert loss is not None
    loss.backward()


def test_pack_weights_compact_binary() -> None:
    rng = np.random.default_rng(2)
    mats = [rng.standard_normal((32, 64)).astype(np.float32) for _ in range(3)]
    emb = rng.standard_normal((128, 32)).astype(np.float32)
    out = pack_weights_compact(mats, embed=emb, linear_mode="binary", embed_mode="int8")
    assert out["linear_bytes"] > 0
    assert out["embed_bytes"] > 0
    assert len(out["linears"]) == 3  # type: ignore[arg-type]


def test_export_l3b2_magic(tmp_path: Path) -> None:
    from scripts.export_weights import MAGIC_BINARY, pack_checkpoint

    cfg = GPTConfig(
        vocab_size=128,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_kv_head=1,
        intermediate_size=128,
        layer_scale_init=1e-4,
    )
    m = GPT(cfg)
    ck = {
        "config": cfg.to_dict(),
        "params": {k: v.detach().cpu() for k, v in m.state_dict().items()},
    }
    pt = tmp_path / "tiny.pt"
    torch.save(ck, pt)
    blob = pack_checkpoint(pt, quant="binary")
    assert blob[:4] == MAGIC_BINARY
    # header has vocab etc.
    V, T, L = struct.unpack_from("<III", blob, 4)
    assert V == 128 and T == 32 and L == 2


def test_binary_infer_ws_under_ternary_same_shape() -> None:
    cfg = GPTConfig(
        vocab_size=2048,
        block_size=256,
        n_layer=23,
        n_head=12,
        n_embd=576,
        n_kv_head=1,
        intermediate_size=1536,
        layer_scale_init=1e-4,
    )
    t = estimate_infer_ws(cfg, quant="ternary", embed_quant="int8")
    b = estimate_infer_ws(cfg, quant="binary", embed_quant="int8")
    assert b.total < t.total
