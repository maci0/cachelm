"""Verifiable tool-policy rewards (GLM5.2 toolsmith-inspired, pure functions).

Use for GRPO-lite / scoring / data filters. No model, no network.

Components (sum clamped to [-1, 1]):
  +format     well-formed single <call>…</call> when a tool is expected
  +schema     tool name in registry and args not empty garbage
  +tool_match selected tool == expected tool
  +arg_match  calc/lookup args match reference (when provided)
  +outcome    final answer contains expected substr (after tool) or no-call clean
  -missed     needed a tool but no call
  -false_call no tool needed but emitted a call
  -unknown    called a tool not in the registry

Format weight can be annealed (early shape, late outcome) via `anneal_format_weight`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ll3m.tools import DEFAULT_TOOLS, parse_call

_CALL_ANY = re.compile(r"<call>\s*([a-zA-Z_][\w]*)", re.I)
_DEGEN = re.compile(r"(?i)(cal){3,}|(call){3,}|calloc")


def anneal_format_weight(progress: float, w0: float = 0.2, w_min: float = 0.0) -> float:
    """Linearly decay format shaping from w0 → w_min as progress ∈ [0, 1]."""
    p = min(max(progress, 0.0), 1.0)
    return w0 * (1.0 - p) + w_min * p


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    format: float = 0.0
    schema: float = 0.0
    tool_match: float = 0.0
    arg_match: float = 0.0
    outcome: float = 0.0
    missed: float = 0.0
    false_call: float = 0.0
    unknown: float = 0.0
    degen: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "format": self.format,
            "schema": self.schema,
            "tool_match": self.tool_match,
            "arg_match": self.arg_match,
            "outcome": self.outcome,
            "missed": self.missed,
            "false_call": self.false_call,
            "unknown": self.unknown,
            "degen": self.degen,
        }


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_completion(
    assistant: str,
    *,
    expect_tool: str | None,
    expect_args: list[str] | None = None,
    expect_substr: list[str] | None = None,
    format_w: float = 0.2,
    registry: set[str] | None = None,
) -> RewardBreakdown:
    """Score one assistant turn (may include <call> and later prose).

    expect_tool:
      - None  → no-call episode (false_call if any <call>)
      - name  → must call that tool (missed if absent)
    """
    reg = registry or set(DEFAULT_TOOLS.keys())
    text = assistant or ""
    degen = -1.0 if _DEGEN.search(text) else 0.0

    parsed = parse_call(text)
    raw_names = [m.group(1).lower() for m in _CALL_ANY.finditer(text)]
    has_call = parsed is not None or bool(raw_names)

    fmt = sch = tmatch = amatch = outcome = missed = false_c = unknown = 0.0

    if expect_tool is None:
        if has_call:
            false_c = -0.5
            # unknown tool spam is worse
            if raw_names and raw_names[0] not in reg:
                unknown = -0.3
        else:
            fmt = format_w  # clean no-call is good form
            if expect_substr and any(s.lower() in text.lower() for s in expect_substr):
                outcome = 0.3
            elif not expect_substr and text.strip() and "<call>" not in text.lower():
                outcome = 0.2
    else:
        want = expect_tool.lower()
        if not has_call:
            missed = -1.0
        else:
            name = (parsed[0] if parsed else raw_names[0]).lower()
            args = list(parsed[1]) if parsed else []
            if name not in reg:
                unknown = -0.5
            else:
                sch = 0.3
            # well-formed tag
            if parsed is not None:
                fmt = format_w
            if name == want:
                tmatch = 0.3
            else:
                tmatch = -0.2
            if expect_args is not None and parsed is not None:
                # normalize string args
                got = [str(a).lower() for a in args]
                exp = [str(a).lower() for a in expect_args]
                if got[: len(exp)] == exp:
                    amatch = 0.2
                elif any(e in got for e in exp):
                    amatch = 0.05
                else:
                    amatch = -0.1
            if expect_substr:
                low = text.lower()
                if any(s.lower() in low for s in expect_substr):
                    outcome = 0.3
                else:
                    outcome = -0.1

    total = _clamp(
        fmt + sch + tmatch + amatch + outcome + missed + false_c + unknown + degen
    )
    return RewardBreakdown(
        total=total,
        format=fmt,
        schema=sch,
        tool_match=tmatch,
        arg_match=amatch,
        outcome=outcome,
        missed=missed,
        false_call=false_c,
        unknown=unknown,
        degen=degen,
    )


def score_calc_episode(
    assistant: str,
    *,
    op: str,
    a: str,
    b: str,
    result: str,
    format_w: float = 0.2,
) -> RewardBreakdown:
    return score_completion(
        assistant,
        expect_tool="calc",
        expect_args=[op, a, b],
        expect_substr=[str(result)],
        format_w=format_w,
    )


def score_lookup_episode(
    assistant: str,
    *,
    key_tokens: list[str],
    value: str,
    format_w: float = 0.2,
) -> RewardBreakdown:
    return score_completion(
        assistant,
        expect_tool="lookup",
        expect_args=key_tokens,
        expect_substr=[value],
        format_w=format_w,
    )


def score_nocall_episode(assistant: str, format_w: float = 0.2) -> RewardBreakdown:
    return score_completion(assistant, expect_tool=None, format_w=format_w)
