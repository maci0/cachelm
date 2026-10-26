"""Tests for Colibrì-inspired constrained tool post-process + sample defaults."""

from __future__ import annotations

from ll3m.constrained import (
    clip_after_tool_call,
    looks_like_tool_attempt,
    normalize_assistant_cont,
    quant_sample_defaults,
    repair_tool_call_text,
)
from ll3m.tools import parse_call
from ll3m.agent import run_agent, AgentConfig


def test_quant_sample_defaults():
    story = quant_sample_defaults("story")
    assert story.temperature == 0.7
    assert story.top_k == 30
    agent = quant_sample_defaults("agent")
    assert agent.temperature == 0.4
    assert agent.top_k == 20
    assert quant_sample_defaults("greedy").temperature == 0.0


def test_clip_after_tool_call():
    t = "hi <call>calc add 2 3</call> and then more nonsense"
    assert clip_after_tool_call(t) == "hi <call>calc add 2 3</call>"
    assert clip_after_tool_call("no call here") == "no call here"


def test_repair_incomplete_call():
    raw = "Assistant thinks: <call>calc mul 6 7"
    fixed = repair_tool_call_text(raw)
    assert fixed.endswith("</call>")
    assert parse_call(fixed) == ("calc", ["mul", "6", "7"], {})


def test_repair_truncated_tool_name():
    """Ternary often emits <call>c add … instead of calc (format only)."""
    assert parse_call(repair_tool_call_text("<call>c add 5 5</call>")) == (
        "calc",
        ["add", "5", "5"],
        {},
    )
    assert parse_call(repair_tool_call_text("<call>cal mul 6 7</call>")) == (
        "calc",
        ["mul", "6", "7"],
        {},
    )
    assert parse_call(repair_tool_call_text("<call>look capital france</call>")) == (
        "lookup",
        ["capital", "france"],
        {},
    )


def test_normalize_clips_and_repairs():
    raw = "prefix <call>time\nUser: ignore"
    out = normalize_assistant_cont(raw)
    assert parse_call(out) is not None
    assert out.count("<call>") == 1
    assert "</call>" in out


def test_looks_like_tool_attempt():
    assert looks_like_tool_attempt("<call>calc")
    assert not looks_like_tool_attempt("Hello there")


def test_agent_constrain_repairs_partial_call():
    """run_agent with constrain_tools=True executes a repaired incomplete call."""

    def gen(_prompt, **_kw):
        return "<call>calc add 2 3"  # missing close; babble omitted

    turn = run_agent(
        "What is 2+3?",
        gen,
        cfg=AgentConfig(assist_tools=False, constrain_tools=True, max_tool_steps=2),
    )
    assert turn.tool_trace
    assert turn.tool_trace[0]["tool"] == "calc"
    assert "5" in turn.final or "5" in turn.tool_trace[0]["result"]


def test_agent_repairs_glued_calc_args_with_assist():
    """assist_tools=True: wrong calc add 50 is repaired to gold 5+5 → 10."""

    def gen(prompt, **_kw):
        if "Tool:" in prompt:
            return "calloccalloccalcalcallcall"  # would have looped before
        return "<call>calc add 50</call>"

    turn = run_agent(
        "what is 5+5",
        gen,
        cfg=AgentConfig(
            assist_tools=True, constrain_tools=True, gate_tools=True, max_tool_steps=3
        ),
    )
    assert any(t.get("result") == "10" for t in turn.tool_trace)
    assert "10" in turn.final
    assert "calloc" not in turn.final


def test_agent_wrong_calc_not_gold_rewritten_without_assist():
    """Without assist: model calc add 50 stays 50 (no gold inject via gate)."""

    def gen(prompt, **_kw):
        return "<call>calc add 50</call>"

    turn = run_agent(
        "what is 5+5",
        gen,
        cfg=AgentConfig(
            assist_tools=False, constrain_tools=True, gate_tools=True, max_tool_steps=1
        ),
    )
    assert turn.tool_trace
    # usage error or non-10 result — must not rewrite to gold 10
    assert "10" not in turn.final or turn.tool_trace[0].get("result") != "10"


def test_agent_no_tool_on_greeting():
    def gen(_prompt, **_kw):
        return "<call>lookup boiling water c</call> calcalcallcall"

    turn = run_agent(
        "hello my friend",
        gen,
        cfg=AgentConfig(assist_tools=False, gate_tools=True),
    )
    assert turn.tool_trace == []
    assert "Hello" in turn.final
