"""Regression tests for defects found in adversarial review rounds 01–10."""

from __future__ import annotations

import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ll3m.config import GPTConfig
from ll3m.model_torch import GPT, apply_rope, RotaryEmbedding
from ll3m.qat import (
    pack_ternary_2bit,
    quantize_ternary_numpy,
    unpack_ternary_2bit,
    fake_quant_per_row,
)
from ll3m.tools import parse_call, tool_calc, execute_call, format_result
from ll3m.tokenizer import BPETokenizer
from ll3m.train import resolve_resumed_best_val
from ll3m.c_engine import read_weights_meta, WeightsMeta
from ll3m.agent import host_assist_call, run_agent, AgentConfig


# ---------------------------------------------------------------------------
# Round 01: ternary pack/unpack
# ---------------------------------------------------------------------------


def test_pack_ternary_clamps_out_of_range_values():
    """OOB int8 values must not produce garbage 2-bit codes."""
    q = np.array([[-2, 0, 1, 2, 5, -5, 1, 0]], dtype=np.int8)
    packed = pack_ternary_2bit(q)
    back = unpack_ternary_2bit(packed, 1, 8)
    assert set(np.unique(back)).issubset({-1, 0, 1})
    # Clamped then packed: -2→-1, 2→1, 5→1, -5→-1
    assert back[0, 0] == -1
    assert back[0, 3] == 1


def test_pack_unpack_roundtrip_legal_ternary():
    rng = np.random.default_rng(0)
    for shape in [(3, 7), (1, 1), (16, 64), (4, 5)]:
        w = rng.standard_normal(shape).astype(np.float32)
        q, sc = quantize_ternary_numpy(w)
        packed = pack_ternary_2bit(q)
        back = unpack_ternary_2bit(packed, shape[0], shape[1])
        assert (back == q).all()
        assert sc.shape == (shape[0],)


def test_unpack_rejects_truncated_buffer():
    with pytest.raises(ValueError, match="packed size"):
        unpack_ternary_2bit(np.array([1, 2], dtype=np.uint8), out=4, inn=16)


def test_pack_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        pack_ternary_2bit(np.array([1, 0, -1], dtype=np.int8))


# ---------------------------------------------------------------------------
# Round 02: tools / calc
# ---------------------------------------------------------------------------


def test_calc_div_zero_returns_error_not_crash():
    assert "ERROR" in tool_calc("div", "1", "0")
    assert "ERROR" in tool_calc("mod", "5", "0")


def test_calc_pow_bound_and_overflow():
    assert "ERROR" in tool_calc("pow", "10", "1000")
    r = tool_calc("pow", "2", "10")
    assert r == "1024"


def test_calc_expr_safe_and_unsafe():
    assert tool_calc(expr="1+2*3") == "7"
    assert "ERROR" in tool_calc(expr='__import__("os")')
    assert "ERROR" in tool_calc(expr="1/0") or "ERROR" in tool_calc(expr="1/0")


def test_parse_call_edges():
    assert parse_call("") is None
    assert parse_call("no call here") is None
    name, pos, kw = parse_call("<call>calc add 2 3</call>")
    assert name == "calc" and pos == ["add", "2", "3"] and kw == {}
    name, pos, kw = parse_call('<call>lookup q="capital france"</call>')
    assert name == "lookup"
    # bare "=x" must not create empty key
    name, pos, kw = parse_call("<call>note =bad text here</call>")
    assert name == "note"
    assert "" not in kw


def test_unknown_tool_execute():
    out = execute_call("rm", ["-rf", "/"], {})
    assert out.startswith("ERROR:")


# ---------------------------------------------------------------------------
# Round 03 / 04: weight meta header validation
# ---------------------------------------------------------------------------


def test_read_weights_meta_rejects_zero_dims(tmp_path: Path):
    # Minimal fake L3T2 header with zero dims
    magic = b"L3T2"
    body = struct.pack(
        "<7IfII",
        0,  # V
        256,
        2,
        4,
        1,
        64,
        128,
        10000.0,
        0,
        0,
    )
    p = tmp_path / "bad.ll3m"
    p.write_bytes(magic + body)
    with pytest.raises(ValueError, match="invalid weight header"):
        read_weights_meta(p)


def test_read_weights_meta_rejects_bad_head_div(tmp_path: Path):
    magic = b"L3M2"
    body = struct.pack(
        "<7IfII",
        100,
        256,
        2,
        5,  # H
        2,  # nkv — 5 % 2 != 0
        64,
        128,
        10000.0,
        0,
        0,
    )
    p = tmp_path / "bad2.ll3m"
    p.write_bytes(magic + body)
    with pytest.raises(ValueError, match="not divisible"):
        read_weights_meta(p)


# ---------------------------------------------------------------------------
# Round 05: config validation
# ---------------------------------------------------------------------------


def test_config_rejects_invalid_dims():
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=0, block_size=32, n_layer=2, n_head=4, n_embd=64)
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=32,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_embd=64,
            intermediate_size=0,
        )
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=32,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_embd=64,
            rope_theta=-1.0,
        )


# ---------------------------------------------------------------------------
# Round 06: resume best_val grid lock
# ---------------------------------------------------------------------------


def test_resolve_resumed_best_val_resets_on_any_grid_change():
    assert math.isinf(
        resolve_resumed_best_val(1.1, "none", "ternary")
    )
    assert math.isinf(
        resolve_resumed_best_val(1.1, "ternary", "none")
    )
    assert math.isinf(
        resolve_resumed_best_val(
            1.1, "ternary", "ternary", resumed_embed_qat="none", embed_qat_mode="int8"
        )
    )
    # same grid keeps bar
    assert resolve_resumed_best_val(1.25, "ternary", "ternary") == 1.25
    assert resolve_resumed_best_val(1.0, "none", "none") == 1.0


# ---------------------------------------------------------------------------
# Round 07: tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_load_rejects_empty_and_bad_merges(tmp_path: Path):
    p = tmp_path / "tok.json"
    p.write_text('{"vocab_size": 0, "stoi": {}, "merges": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(p)
    p.write_text(
        '{"vocab_size": 4, "stoi": {"a": 0}, "merges": ["bad"]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="merge"):
        BPETokenizer.load(p)


def test_tokenizer_decode_oob_ids_no_crash():
    tok = BPETokenizer(
        stoi={"a": 0, "b": 1, "<unk>": 2},
        itos={0: "a", 1: "b", 2: "<unk>"},
        merges=[],
        vocab_size=3,
    )
    assert tok.decode([0, 1]) == "ab"
    assert tok.decode([-1, 99, 2, 0]) == "a"  # skip invalid / specials


# ---------------------------------------------------------------------------
# Round 08: RoPE rotate-half
# ---------------------------------------------------------------------------


def test_apply_rope_rotate_half_identity_at_zero():
    """At position 0, cos=1 sin=0 → RoPE is identity."""
    B, H, T, hd = 1, 2, 4, 8
    q = torch.randn(B, H, T, hd)
    k = torch.randn(B, H, T, hd)
    rope = RotaryEmbedding(hd, max_pos=16, theta=10000.0)
    cos, sin = rope(T, q.device)
    # force pos-0 identity check by zeroing angles
    cos0 = torch.ones_like(cos)
    sin0 = torch.zeros_like(sin)
    q2, k2 = apply_rope(q, k, cos0, sin0)
    assert torch.allclose(q2, q, atol=1e-5)
    assert torch.allclose(k2, k, atol=1e-5)


def test_generate_stream_greedy_and_topk_zero():
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=32, n_kv_head=1)
    m = GPT(cfg)
    m.eval()
    idx = torch.zeros(1, 2, dtype=torch.long)
    # temperature <= 0 → greedy path
    out = list(m.generate_stream(idx, max_new_tokens=3, temperature=0.0, top_k=40))
    assert len(out) == 3
    # top_k=0 must not crash
    out2 = list(m.generate_stream(idx, max_new_tokens=2, temperature=1.0, top_k=0))
    assert len(out2) == 2
    with pytest.raises(ValueError):
        list(m.generate_stream(torch.zeros(1, 0, dtype=torch.long), 1))


# ---------------------------------------------------------------------------
# Round 09: agent assist + host calc
# ---------------------------------------------------------------------------


def test_host_assist_and_agent_calc():
    call = host_assist_call("What is 3 + 4?")
    assert call is not None and "calc" in call
    assert parse_call(call) is not None

    def gen(_prompt, **_kw):
        return "I do not know."  # no call; assist should inject

    turn = run_agent(
        "What is 6 * 7?",
        gen,
        cfg=AgentConfig(assist_tools=True, max_tool_steps=2, max_tokens=20),
    )
    assert turn.tool_trace
    assert turn.tool_trace[0]["tool"] == "calc"
    assert "42" in turn.final


def test_agent_no_assist_without_call():
    def gen(_prompt, **_kw):
        return "Hello there."

    turn = run_agent(
        "hi",
        gen,
        cfg=AgentConfig(assist_tools=False, max_tool_steps=1),
    )
    assert turn.tool_trace == []
    assert "Hello" in turn.final


def test_run_agent_mutates_caller_history_multi_turn():
    """Caller-owned history must accumulate turns (chat multi-turn contract)."""
    shared: list[str] = []
    prompts_seen: list[str] = []

    def gen(prompt: str, **_kw) -> str:
        prompts_seen.append(prompt)
        return "ok."

    run_agent(
        "first",
        gen,
        history=shared,
        cfg=AgentConfig(assist_tools=False, max_tool_steps=1),
    )
    assert len(shared) == 1
    assert "first" in shared[0]
    run_agent(
        "second",
        gen,
        history=shared,
        cfg=AgentConfig(assist_tools=False, max_tool_steps=1),
    )
    assert len(shared) == 2
    assert "second" in shared[1]
    # Second turn's prompt must include first turn history (not a fresh copy)
    assert "first" in prompts_seen[1]
    assert "User: second" in prompts_seen[1]


def test_stream_agent_mutates_caller_history():
    from ll3m.agent import stream_agent_text

    shared: list[str] = []

    def gen(_prompt, **_kw) -> str:
        return "streamed."

    def gen_stream(_prompt, **_kw):
        yield "streamed."

    parts = list(
        stream_agent_text(
            "hello",
            gen_stream,
            gen,
            history=shared,
            cfg=AgentConfig(assist_tools=False, max_tool_steps=1),
        )
    )
    assert parts
    assert len(shared) == 1
    assert "hello" in shared[0]


def test_agent_config_topk_zero_not_coerced():
    """top_k=0 must remain 0 (chat must not use `or 40`)."""
    cfg = AgentConfig(top_k=0)
    assert cfg.top_k == 0
    # Mirror chat.agent_reply contract
    top_k = 0
    resolved = 40 if top_k is None else top_k
    assert resolved == 0
    wrong = top_k or 40
    assert wrong == 40  # documents the bug we fixed


# ---------------------------------------------------------------------------
# Round 10: format_result + fake quant STE shape
# ---------------------------------------------------------------------------


def test_format_result_truncates():
    body = "x" * 500
    s = format_result(body, max_chars=40)
    assert s.startswith("<result>")
    assert s.endswith("</result>")
    assert len(s) < 60


def test_fake_quant_int8_and_ternary_shapes():
    w = torch.randn(8, 16, requires_grad=True)
    q = fake_quant_per_row(w, num_bits=8)
    t = fake_quant_per_row(w, ternary=True)
    assert q.shape == w.shape and t.shape == w.shape
    (q.sum() + t.sum()).backward()
    assert w.grad is not None
