"""Budget / architecture gates for the v2 product path."""

from __future__ import annotations

from ll3m.cache_info import budget_bytes, probe_cache
from ll3m.config import GPTConfig
from ll3m.model_torch import GPT, choose_config_infer, estimate_budget, estimate_infer_ws


def test_probe_cache_has_l3() -> None:
    hier = probe_cache()
    assert hier.l3_bytes >= 1024 * 1024
    assert hier.l3_total_bytes >= hier.l3_bytes


def test_chip_budget_positive() -> None:
    hier = probe_cache()
    b = budget_bytes(hier, fraction=0.90, scope="chip")
    assert b > 0
    assert b <= hier.l3_total_bytes


def test_product_shape_param_count() -> None:
    """Frozen product shape L23 d576 MQA mid1536."""
    cfg = GPTConfig(
        vocab_size=2048,
        block_size=256,
        n_layer=23,
        n_head=12,
        n_embd=576,
        n_kv_head=1,
        intermediate_size=1536,
        layer_scale_init=1e-4,
        dropout=0.1,
    )
    assert cfg.n_embd % cfg.n_head == 0
    assert cfg.arch == "v2"
    model = GPT(cfg)
    torch_n = sum(p.numel() for p in model.parameters())
    assert torch_n == cfg.n_params()
    # ~78M class
    assert 70_000_000 < torch_n < 90_000_000


def test_qat_fake_quant_shapes() -> None:
    import numpy as np
    import torch
    from ll3m.qat import (
        enable_qat_on_model,
        fake_quant_per_row,
        pack_ternary_2bit,
        quantize_ternary_numpy,
        unpack_ternary_2bit,
    )

    w = torch.randn(16, 32)
    q = fake_quant_per_row(w, num_bits=8)
    assert q.shape == w.shape
    t = fake_quant_per_row(w, ternary=True)
    assert t.shape == w.shape
    wn = w.detach().numpy()
    qt, sc = quantize_ternary_numpy(wn)
    assert qt.shape == wn.shape
    assert set(np.unique(qt)).issubset({-1, 0, 1})
    packed = pack_ternary_2bit(qt)
    back = unpack_ternary_2bit(packed, qt.shape[0], qt.shape[1])
    assert (back == qt).all()

    m = GPT(GPTConfig(64, 32, 2, 4, 64, n_kv_head=2))
    n = enable_qat_on_model(m, ternary=True)
    assert n > 0
    x = torch.randint(0, 64, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape[-1] == 64
    assert loss is not None
    loss.backward()


def test_v2_param_count_matches_torch() -> None:
    cfg = GPTConfig(
        vocab_size=2048,
        block_size=256,
        n_layer=8,
        n_head=8,
        n_embd=256,
        n_kv_head=2,
        dropout=0.1,
    )
    model = GPT(cfg)
    torch_n = sum(p.numel() for p in model.parameters())
    assert torch_n == cfg.n_params()
    inf = estimate_infer_ws(cfg)
    assert inf.total > 0
    assert inf.n_params == torch_n


def test_binary_linear_pack_smaller_ws_than_ternary() -> None:
    """Capacity-search binary (1-bit) Linears must estimate under ternary (2-bit)."""
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
    tern = estimate_infer_ws(cfg, quant="ternary", embed_quant="int8")
    binary = estimate_infer_ws(cfg, quant="binary", embed_quant="int8")
    assert binary.weight_bytes < tern.weight_bytes
    assert binary.total < tern.total
    # ~9–11 MiB free on product shape (pack only; acts unchanged)
    saved = tern.total - binary.total
    assert 8 * 1024 * 1024 < saved < 12 * 1024 * 1024


def test_choose_config_infer_respects_budget() -> None:
    hier = probe_cache()
    budget = int(hier.l3_bytes * 0.95)
    cfg, inf = choose_config_infer(2048, budget, block_size=256)
    assert cfg.n_embd % cfg.n_head == 0
    assert inf.total <= budget
    est = estimate_budget(cfg, batch=8, optimizer="lion")
    assert est.n_params == cfg.n_params()
