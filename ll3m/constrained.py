"""Constrained tool-call post-processing + quant-aware sampling defaults.

Inspired by Colibrì (JustVugg/colibri) grammar-draft / quality-preserving serve:
  - Never silently change model precision; clamp *sampling* under quant noise.
  - For tools, clip after a complete <call>…</call> (no babble past the tag).
  - Repair common SLM format failures before host execute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ll3m.tools import CALL_CLOSE, CALL_OPEN, DEFAULT_TOOLS, parse_call

# Incomplete open tag through end of string (DOTALL)
_OPEN_CALL_RE = re.compile(
    r"<call>\s*([a-zA-Z_][\w]*)\s*(.*)$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class SampleDefaults:
    """Recommended temperature / top_k for a serve mode under ternary weights."""

    temperature: float
    top_k: int
    note: str


def quant_sample_defaults(mode: str = "story") -> SampleDefaults:
    """Colibrì lesson: high temp samples quantization noise in the tail.

    Official int4 defaults there: temp 0.7 / top-p 0.90 (not 1.0 / 0.95).
    We use top_k (C engine) as the concentration knob.
    """
    m = (mode or "story").lower()
    if m in ("agent", "tool", "tools"):
        return SampleDefaults(
            temperature=0.4,
            top_k=20,
            note="agent/tools: low temp + tight top_k for format reliability",
        )
    if m in ("greedy", "eval"):
        return SampleDefaults(
            temperature=0.0,
            top_k=1,
            note="greedy decode",
        )
    # story / complete
    return SampleDefaults(
        temperature=0.7,
        top_k=30,
        note="story under ternary: calmer than 0.8/40",
    )


def clip_after_tool_call(text: str) -> str:
    """Keep text through first complete </call>; drop trailing babble.

    If no complete call, return text unchanged (caller may repair).
    """
    if not text:
        return text
    low = text.lower()
    close = low.find(CALL_CLOSE)
    if close < 0:
        return text
    end = close + len(CALL_CLOSE)
    return text[:end]


def repair_tool_call_text(
    text: str,
    *,
    tools: dict | None = None,
) -> str:
    """Normalize an assistant turn that may contain a partial tool call.

    - Prefer first well-formed <call>…</call>
    - If open tag without close: close it when name is a known tool
    - Drop trailing junk after a complete call
    - Unknown tool names: leave text (host execute returns ERROR)
    """
    if not text:
        return text
    tools = tools or DEFAULT_TOOLS
    tool_names = {t.lower() for t in tools}

    def _fix_tool_name(name: str, rest: str) -> str:
        """Truncated tool names under ternary noise (format only, not gold args)."""
        n = (name or "").lower()
        if n in tool_names:
            return n
        if n in {"c", "cal", "ca"} and re.match(
            r"(?i)^(add|sub|mul|div|\+|\-|\*|/|x)\b", rest or ""
        ):
            return "calc"
        if n in {"look", "looku", "l"} and rest:
            return "lookup"
        if n in {"ti", "tim"} and not rest:
            return "time"
        return n

    # Complete call: maybe rewrite truncated name, then clip babble
    parsed = parse_call(text)
    if parsed is not None:
        pname, pargs, pkwargs = parsed
        rest_bits = list(pargs) + [f"{k}={v}" for k, v in pkwargs.items()]
        rest = " ".join(str(x) for x in rest_bits)
        fixed = _fix_tool_name(str(pname), rest)
        if fixed != str(pname).lower():
            body = f"{CALL_OPEN}{fixed}"
            if rest:
                body += f" {rest}"
            body += CALL_CLOSE
            # keep any short prefix before the call
            m = _OPEN_CALL_RE.search(text)
            prefix = text[: m.start()] if m else ""
            return prefix + body
        return clip_after_tool_call(text)

    m = _OPEN_CALL_RE.search(text)
    if not m:
        return text
    name = m.group(1).lower()
    rest = (m.group(2) or "").strip()
    # Strip accidental nested close / trailing assistant markers
    rest = re.split(r"</call>|\nUser:|\nTool:", rest, maxsplit=1)[0].strip()
    name = _fix_tool_name(name, rest)
    # Rebuild a single clean call from the open span start
    prefix = text[: m.start()]
    body = f"{CALL_OPEN}{name}"
    if rest:
        body += f" {rest}"
    body += CALL_CLOSE
    return prefix + body


def normalize_assistant_cont(
    text: str,
    *,
    tools: dict | None = None,
) -> str:
    """Full post-step normalize for agent generate output."""
    t = text or ""
    t = repair_tool_call_text(t, tools=tools)
    t = clip_after_tool_call(t)
    return t


def looks_like_tool_attempt(text: str) -> bool:
    """True if the model started a tool tag (even incomplete)."""
    if not text:
        return False
    return CALL_OPEN in text.lower() or "<call" in text.lower()
