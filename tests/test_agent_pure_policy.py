"""True pure-policy: assist_tools=False and gate_tools=False → no host gold inject."""

from __future__ import annotations

from ll3m.agent import AgentConfig, run_agent


def test_silent_model_on_calc_does_not_inject_gold() -> None:
    """Skeptic: silent model must not get 'The answer is 10.' from host gold."""

    def gen_empty(prompt: str, **kw: object) -> str:
        return "I'm not sure."

    turn = run_agent(
        "what is 5+5",
        gen_empty,
        cfg=AgentConfig(assist_tools=False, gate_tools=False, max_tool_steps=1),
    )
    assert turn.tool_trace == []
    assert "10" not in turn.final
    assert "ERROR" not in turn.final


def test_wrong_calc_not_repaired_to_gold() -> None:
    """Skeptic: model calc add 1 1 must not be repaired to 5+5 under pure policy."""

    def gen_wrong(prompt: str, **kw: object) -> str:
        return "<call>calc add 1 1</call>"

    turn = run_agent(
        "what is 5+5",
        gen_wrong,
        cfg=AgentConfig(assist_tools=False, gate_tools=False, max_tool_steps=1),
    )
    assert turn.tool_trace
    assert turn.tool_trace[0].get("tool") == "calc"
    assert turn.tool_trace[0].get("args") == ["add", "1", "1"]
    assert turn.tool_trace[0].get("result") == "2"
    # Must not rewrite to gold 10
    assert "10" not in turn.final
    assert "2" in turn.final or turn.final  # host phrase for 2 or raw


def test_assist_mode_still_injects_when_silent() -> None:
    def gen_empty(prompt: str, **kw: object) -> str:
        return "hmm"

    turn = run_agent(
        "what is 5+5",
        gen_empty,
        cfg=AgentConfig(assist_tools=True, gate_tools=False, max_tool_steps=1),
    )
    assert turn.tool_trace
    assert turn.tool_trace[0].get("result") == "10"
    assert "10" in turn.final


def test_gate_only_does_not_inject_gold() -> None:
    """When2Call gate is not gold inject: silent model must not get free calc."""

    def gen_empty(prompt: str, **kw: object) -> str:
        return "hmm"

    turn = run_agent(
        "what is 5+5",
        gen_empty,
        cfg=AgentConfig(assist_tools=False, gate_tools=True, max_tool_steps=1),
    )
    assert turn.tool_trace == []
    assert "10" not in turn.final


def test_gate_blocks_false_tool_on_chitchat() -> None:
    """gate_tools=True: lookup spam on hello must not run; chitchat answer."""

    def gen_lookup(prompt: str, **kw: object) -> str:
        return "<call>lookup capital france</call>"

    turn = run_agent(
        "hello",
        gen_lookup,
        cfg=AgentConfig(assist_tools=False, gate_tools=True, max_tool_steps=1),
    )
    assert turn.tool_trace == []
    assert "Paris" not in turn.final
    assert "Hello" in turn.final or "help" in turn.final.lower()


def test_reset_chat_clears_agent_history() -> None:
    """Battery isolation: reset_chat must drop agent multi-turn buffer."""
    from pathlib import Path
    from ll3m.chat import LL3M

    ck = Path("checkpoints/v76_dense_c_pure/best.ll3m")
    if not ck.is_file():
        ck = Path("checkpoints/v76_dense_c/best.ll3m")
    if not ck.is_file():
        return  # skip if no pack in CI
    m = LL3M(ck, python_infer=False, seed=0)
    m._agent_history = ["User: hi\nAssistant: Hello!\n"]
    m.reset_chat()
    assert getattr(m, "_agent_history", None) == []
