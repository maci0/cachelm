"""Agent pure-policy: false tool calls must not surface ERROR spam."""

from __future__ import annotations

from ll3m.agent import AgentConfig, run_agent


def test_false_call_on_hello_returns_chitchat() -> None:
    def gen_spam(prompt: str, **kw: object) -> str:
        return "<call>the weather</call>"

    turn = run_agent(
        "hello",
        gen_spam,
        cfg=AgentConfig(assist_tools=False, gate_tools=False, max_tool_steps=1),
    )
    assert "ERROR" not in turn.final
    assert "Hello" in turn.final or "help" in turn.final.lower()
    assert turn.tool_trace
    assert str(turn.tool_trace[0].get("result", "")).startswith("ERROR:")


def test_calc_call_still_executes() -> None:
    def gen_calc(prompt: str, **kw: object) -> str:
        return "<call>calc add 5 5</call>"

    turn = run_agent(
        "what is 5+5",
        gen_calc,
        cfg=AgentConfig(assist_tools=False, gate_tools=False, max_tool_steps=1),
    )
    assert turn.tool_trace
    assert turn.tool_trace[0].get("tool") == "calc"
    assert turn.tool_trace[0].get("args") == ["add", "5", "5"]
    assert turn.tool_trace[0].get("result") == "10"
    assert "10" in turn.final
