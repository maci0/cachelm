"""Unit tests for ll3m.rewards (verifiable tool-policy scores)."""

from __future__ import annotations

from ll3m.rewards import (
    anneal_format_weight,
    score_calc_episode,
    score_completion,
    score_nocall_episode,
)


def test_anneal_format_weight() -> None:
    assert anneal_format_weight(0.0) == 0.2
    assert anneal_format_weight(1.0) == 0.0
    assert abs(anneal_format_weight(0.5) - 0.1) < 1e-9


def test_nocall_clean() -> None:
    r = score_nocall_episode("Hello! How can I help?")
    assert r.false_call == 0.0
    assert r.missed == 0.0
    assert r.total > 0


def test_nocall_false_call_penalized() -> None:
    r = score_nocall_episode("Hello <call>lookup capital france</call>")
    assert r.false_call < 0
    assert r.total < 0


def test_calc_good() -> None:
    text = (
        "<call>calc mul 3 4</call>\n"
        "Tool: <result>12</result>\n"
        "3 times 4 is 12."
    )
    r = score_calc_episode(text, op="mul", a="3", b="4", result="12")
    assert r.missed == 0.0
    assert r.tool_match > 0
    assert r.total > 0.3


def test_calc_missed() -> None:
    r = score_calc_episode("I think it is 12.", op="mul", a="3", b="4", result="12")
    assert r.missed == -1.0
    assert r.total < 0


def test_unknown_tool() -> None:
    r = score_completion(
        "<call>the weather</call>",
        expect_tool=None,
    )
    assert r.false_call < 0 or r.unknown < 0
    assert r.total < 0


def test_wrong_tool() -> None:
    r = score_completion(
        "<call>time</call>",
        expect_tool="calc",
        expect_args=["add", "1", "1"],
        expect_substr=["2"],
    )
    assert r.tool_match < 0
