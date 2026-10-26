"""Product dense ternary candidate: v75 freeze shape, WS gate, modern v2 path.

Canonical freeze: checkpoints/v75_freeze.json (L23×d576 MQA, peri off, ls 1e-4).
The older L18×d320 / peri-on candidate is superseded; do not restore it here.
"""

from __future__ import annotations

import torch

from ll3m.cache_info import probe_cache
from ll3m.config import GPTConfig
from ll3m.model_torch import GPT, estimate_infer_ws
from ll3m.qat import enable_qat_on_model, fake_quant_per_row

# Product freeze (matches checkpoints/v75_freeze.json and AGENTS.md).
PRODUCT_N_PARAMS = 78_815_712


def _product_cfg(**kw) -> GPTConfig:
    base = dict(
        vocab_size=2048,
        block_size=256,
        n_layer=23,
        n_head=12,
        n_embd=576,
        n_kv_head=1,
        intermediate_size=1536,
        peri_ln=False,
        layer_scale_init=1e-4,
        dropout=0.1,
    )
    base.update(kw)
    return GPTConfig(**base)


def test_product_ternary_ws_fits_ccd_gate() -> None:
    """Serve claim: ternary working set under 95% of one L3 CCD."""
    hier = probe_cache()
    gate = int(hier.l3_bytes * 0.95)
    cfg = _product_cfg()
    # product pack: ternary Linears + int8 embeds (must pass quant=ternary)
    tern = estimate_infer_ws(cfg, quant="ternary", embed_quant="int8")
    assert tern.total <= gate
    assert tern.n_params == cfg.n_params() == PRODUCT_N_PARAMS
    # freeze-class fill: design ~30.22 MiB on 32 MiB CCD hosts
    assert tern.total / (1024 * 1024) < 31.0
    # ternary denser than int8 Linears for same shape
    i8 = estimate_infer_ws(cfg, quant="int8", embed_quant="f32")
    assert tern.total < i8.total
    # compact embeds beat f32 embeds under same Linear quant
    fat = estimate_infer_ws(cfg, quant="ternary", embed_quant="f32")
    assert tern.total < fat.total
    # ternary embeds denser still
    emb_t = estimate_infer_ws(cfg, quant="ternary", embed_quant="ternary")
    assert emb_t.total <= tern.total


def test_pack_weights_compact_ternary_and_embed_int8() -> None:
    """Real pack helpers produce 2-bit Linear + int8 embed buffers."""
    import numpy as np
    from ll3m.qat import pack_weights_compact, unpack_ternary_2bit

    w = np.random.randn(32, 64).astype(np.float32)
    e = np.random.randn(128, 64).astype(np.float32)
    packed = pack_weights_compact([w], embed=e, linear_mode="ternary", embed_mode="int8")
    assert packed["linear_bytes"] < w.nbytes
    assert packed["embed_bytes"] < e.nbytes
    # roundtrip ternary pack
    q = unpack_ternary_2bit(packed["linears"][0]["q_packed"], 32, 64)
    assert q.shape == (32, 64)
    assert set(np.unique(q)).issubset({-1, 0, 1})
    eq = packed["embed"]["q"]
    assert eq.dtype == np.int8 and eq.shape == e.shape


def test_product_param_count_matches_module() -> None:
    cfg = _product_cfg()
    m = GPT(cfg)
    assert m.n_params() == cfg.n_params()
    assert m.n_params() == PRODUCT_N_PARAMS
    assert cfg.peri_ln is False
    assert cfg.layer_scale_init == 1e-4
    assert cfg.kv_head == 1
    assert cfg.intermediate == 1536


def test_peri_ln_and_layer_scale_increase_params() -> None:
    plain = GPTConfig(
        vocab_size=2048, block_size=64, n_layer=2, n_head=4, n_embd=64,
        n_kv_head=2,
    )
    fancy = GPTConfig(
        vocab_size=2048, block_size=64, n_layer=2, n_head=4, n_embd=64,
        n_kv_head=2, peri_ln=True, layer_scale_init=1e-2,
    )
    assert fancy.n_params() > plain.n_params()
    m = GPT(fancy)
    assert m.n_params() == fancy.n_params()
    # residual scales exist
    assert m.transformer.h[0].ls_1 is not None
    assert m.transformer.h[0].post_attn is not None


def test_mtp_and_z_loss_train_path() -> None:
    """Shared-head MTP + z-loss are real train-time paths on shipped GPT.forward."""
    cfg = GPTConfig(
        vocab_size=128, block_size=32, n_layer=2, n_head=4, n_embd=64,
        n_kv_head=2, peri_ln=True, layer_scale_init=1e-2,
    )
    m = GPT(cfg)
    x = torch.randint(0, 128, (2, 24))
    y = torch.randint(0, 128, (2, 24))
    _, loss0 = m(x, y, z_loss_coeff=0.0, mtp_weight=0.0)
    _, loss1 = m(x, y, z_loss_coeff=1e-4, mtp_weight=0.1, mtp_depth=2)
    assert loss0 is not None and loss1 is not None
    # aux terms should not crash and typically change the scalar
    loss1.backward()
    assert any(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_partial_rope_and_softcap_forward() -> None:
    cfg = GPTConfig(
        vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=64,
        n_kv_head=2, rope_partial=0.5, logit_softcap=30.0,
    )
    assert cfg.rope_dims == 8  # half of head_dim 16, even
    m = GPT(cfg)
    x = torch.randint(0, 64, (1, 16))
    logits, loss = m(x, x)
    assert logits.shape == (1, 16, 64)
    assert loss is not None
    # softcap bounds logits roughly in (-cap, cap)
    assert float(logits.detach().abs().max()) <= 30.0 + 1e-3


def test_resolve_resumed_best_val_resets_on_qat_mode_change() -> None:
    """Shipped early-stop helper: fp32 bar must not block ternary QAT best.pt."""
    from ll3m.train import resolve_resumed_best_val

    # fp32 ckpt → ternary QAT: reset
    assert resolve_resumed_best_val(1.39, "none", "ternary") == float("inf")
    assert resolve_resumed_best_val(1.39, None, "ternary") == float("inf")  # type: ignore[arg-type]
    # same ternary → keep
    assert resolve_resumed_best_val(1.62, "ternary", "ternary") == 1.62
    # no QAT → keep
    assert resolve_resumed_best_val(1.39, "none", "none") == 1.39
    # int8 → ternary still reset
    assert resolve_resumed_best_val(1.5, "int8", "ternary") == float("inf")
    # embed qat change also resets
    assert (
        resolve_resumed_best_val(
            1.55, "ternary", "ternary", resumed_embed_qat="none", embed_qat_mode="ternary"
        )
        == float("inf")
    )
    assert (
        resolve_resumed_best_val(
            1.55, "ternary", "ternary", resumed_embed_qat="int8", embed_qat_mode="int8"
        )
        == 1.55
    )


def test_embed_qat_forward_ste() -> None:
    """Embed ternary/int8 QAT path runs on real GPT.forward with STE."""
    from ll3m.qat import enable_embed_qat_on_model, enable_qat_on_model

    cfg = GPTConfig(
        vocab_size=128, block_size=32, n_layer=2, n_head=4, n_embd=64,
        n_kv_head=2,
    )
    m = GPT(cfg)
    enable_qat_on_model(m, ternary=True)
    enable_embed_qat_on_model(m, mode="ternary")
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    _, loss = m(x, y)
    assert loss is not None
    loss.backward()
    assert m.transformer.wte.weight.grad is not None


def test_ternary_qat_on_product_block() -> None:
    """QAT patches real Linear modules; forward+backward work under ternary STE."""
    cfg = GPTConfig(
        vocab_size=128, block_size=32, n_layer=2, n_head=4, n_embd=64,
        n_kv_head=2, peri_ln=True, layer_scale_init=1e-2,
    )
    m = GPT(cfg)
    n = enable_qat_on_model(m, ternary=True)
    assert n >= 1
    w = m.transformer.h[0].attn.q_proj.weight
    q = fake_quant_per_row(w, ternary=True)
    assert q.shape == w.shape
    x = torch.randint(0, 128, (2, 16))
    logits, loss = m(x, x, z_loss_coeff=1e-4, mtp_weight=0.1)
    assert logits.shape[-1] == 128
    loss.backward()
