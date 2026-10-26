"""Host agent loop: model emits <call>…</call>; host runs tools and continues.

Generation always goes through LL3M (C by default). Tools never run in-model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from ll3m.constrained import normalize_assistant_cont
from ll3m.tools import (
    CALL_CLOSE,
    DEFAULT_TOOLS,
    ToolSpec,
    _norm_key,
    execute_call,
    extract_call_span,
    format_result,
    get_kb,
    parse_call,
    tools_brief,
)


GenerateFn = Callable[..., str]  # prompt -> continuation text


@dataclass
class AgentConfig:
    max_tool_steps: int = 1  # SLMs: one tool then answer (no multi-hop babble)
    max_tokens: int = 80
    # Colibrì / ternary: low temp + tighter top_k (see quant_sample_defaults)
    temperature: float = 0.4
    top_k: int = 20
    max_result_chars: int = 360
    # Product default: no gold inject. Model must emit <call> for calc/lookup.
    # Set True only for bootstrap / debug repair of bad calc args.
    assist_tools: bool = False
    # Clip/repair <call>…</call> after each generate (constrained post-process)
    constrain_tools: bool = True
    # Product default: When2Call gate on (block tools on chitchat). Pure stress
    # test uses gate_tools=False (--pure-policy). See docs/DENSE_CAPABILITY_PLAN.md.
    gate_tools: bool = True

    def host_may_inject(self) -> bool:
        """Gold inject / calc arg repair only when assist_tools is on.

        gate_tools alone: block tools on chitchat, and rewrite a *wrong*
        tool name when host intent is known (e.g. lookup→time). Does not
        invent a call when the model is silent (that is assist).
        Pure policy: assist=False and gate=False.
        """
        return bool(self.assist_tools)

@dataclass
class AgentTurn:
    user: str
    transcript: str
    final: str
    tool_trace: list[dict] = field(default_factory=list)
    # Host-gate interventions (empty when gate_tools=False).
    # kind: block_false_call | rewrite_tool | pass_through
    gate_events: list[dict] = field(default_factory=list)
    model_emitted_call: bool = False
    gold_needs_tool: bool = False


def build_system(tools: dict[str, ToolSpec] | None = None) -> str:
    return (
        "You are a small helper. Use tools for facts and math. "
        "Answer briefly in plain English.\n"
        + tools_brief(tools)
    )


def build_prompt(
    user: str,
    history: list[str],
    *,
    system: str | None = None,
    tools: dict[str, ToolSpec] | None = None,
) -> str:
    parts = [(system or build_system(tools)).rstrip(), ""]
    for h in history:
        parts.append(h.rstrip())
    parts.append(f"User: {user}")
    parts.append("Assistant:")
    return "\n".join(parts) + " "


_CALC_Q = re.compile(
    r"(?i)\b(?:what\s+is|what's|calculate|compute)?\s*"
    r"(-?\d+(?:\.\d+)?)\s*([\+\-\*/x×]|plus|minus|times|over|divided by)\s*"
    r"(-?\d+(?:\.\d+)?)\s*\??\s*$"
)
_TIME_Q = re.compile(r"(?i)\b(what\s+time|current\s+time|utc|what\s+day|date\s+today)\b")
_LOOKUP_Q = re.compile(
    r"(?i)\b(capital\s+of\s+(\w+(?:\s+\w+){0,3})|what\s+is\s+pi|water\s+formula)\b"
)
_WIKI_Q = re.compile(
    r"(?i)\b(?:who\s+is|what\s+is|who\s+was|tell\s+me\s+about)\s+(.+?)\s*\??\s*$"
)
_GREET_Q = re.compile(
    r"(?i)^(hi|hello|hey|howdy|good\s+(morning|afternoon|evening)|yo)\b"
)


def chitchat_reply(user: str) -> str | None:
    """Short no-tool reply for greetings / empty chat."""
    u = user.strip()
    if not u:
        return "I'm here. Ask a question when you're ready."
    if _GREET_Q.search(u):
        return "Hello! How can I help?"
    if re.search(r"(?i)\b(thanks|thank you|thx)\b", u) and len(u) < 40:
        return "You're welcome."
    if re.search(r"(?i)\b(bye|goodbye|see you)\b", u) and len(u) < 40:
        return "Goodbye!"
    if re.search(r"(?i)^how are you", u):
        return "I'm doing well. How can I help?"
    if re.search(r"(?i)who are you|what('?s| is) your name", u):
        return "I'm a small helper. I can do math and look up simple facts."
    return None


def host_assist_call(user: str) -> str | None:
    """Optional bootstrap: invent a tool call the model should have made."""
    m = _CALC_Q.search(user.strip())
    if m:
        a, op_s, b = m.group(1), m.group(2).lower(), m.group(3)
        op_map = {
            "+": "add",
            "plus": "add",
            "-": "sub",
            "minus": "sub",
            "*": "mul",
            "x": "mul",
            "×": "mul",
            "times": "mul",
            "/": "div",
            "over": "div",
            "divided by": "div",
        }
        op = op_map.get(op_s, "mul")
        return f"<call>calc {op} {a} {b}</call>"
    if _TIME_Q.search(user):
        return "<call>time</call>"
    # Closed-KB-backed lookup assists: skip invent when set_kb({}) emptied the table
    # so ablations cannot hit a stale imported _KB alias or invent missing facts.
    kb = get_kb()
    m = _LOOKUP_Q.search(user)
    if m and kb:
        if m.group(2):
            return f"<call>lookup capital {m.group(2).strip()}</call>"
        if re.search(r"(?i)\bpi\b", user):
            return "<call>lookup pi</call>"
        if re.search(r"(?i)water\s*formula|formula\s+of\s+water", user):
            return "<call>lookup water formula</call>"
    # Direct / word-bounded closed-KB keys (Engram-lite). Avoid "pi" ⊂ "pizza".
    # Expanded KB (dense capability P0) is covered here without new regexes.
    # Always use get_kb() / live kb (never bind _KB by name).
    uk = _norm_key(user)
    uk = re.sub(r"^(what\s+is|what's|whats|tell\s+me|define)\s+", "", uk).strip(" ?.")
    if uk in kb:
        return f"<call>lookup {uk}</call>"
    for k in sorted(kb.keys(), key=len, reverse=True):
        if re.search(rf"(?<![\w]){re.escape(k)}(?![\w])", uk):
            return f"<call>lookup {k}</call>"
    m = _WIKI_Q.search(user.strip())
    if m:
        topic = m.group(1).strip().strip("\"'")
        # Avoid "what is 5+5" already handled; skip tiny/non-entity topics
        if 2 <= len(topic) <= 60 and not topic.lower().startswith(
            ("the time", "2+", "2 +")
        ):
            # Don't wiki pure greetings / chitchat / identity smalltalk
            tl = topic.lower()
            if tl in ("your name", "you", "this", "that", "up", "this called"):
                return None
            if not _GREET_Q.search(topic) and not re.fullmatch(r"[\d\s+\-*/.]+", topic):
                return f"<call>wiki {topic}</call>"
    return None


def _format_call_span(name: str, args: list[str], kwargs: dict[str, str]) -> str:
    body = name
    if args:
        body += " " + " ".join(args)
    if kwargs:
        body += "".join(f" {k}={v}" for k, v in kwargs.items())
    return f"<call>{body}</call>"


def _finalize_no_tool_text(user: str, cont: str) -> str:
    """Strip tool tags; greetings get a clean host line instead of empty/babble."""
    final = _CALL_RE_strip(cont).strip()
    if not final or _looks_degenerate(final):
        return chitchat_reply(user) or "I'm not sure. Try a math question or a simple fact lookup."
    return final


def _try_host_repair_call(
    user: str,
    name: str,
    args: list[str],
    kwargs: dict[str, str],
    result_body: str,
) -> tuple[str, list[str], dict[str, str]] | None:
    """If model call failed, recover a known-good call from the user text.

    SLMs often emit ``calc add 50`` for ``5+5`` (args glued). Host regex
    still knows the intent; re-run once with the assisted call.
    """
    if not result_body.startswith("ERROR:"):
        return None
    assisted = host_assist_call(user)
    if not assisted:
        return None
    fixed = parse_call(assisted)
    if fixed is None:
        return None
    # Skip if same broken call (avoid infinite repair)
    fn, fa, fk = fixed
    if fn == name and fa == list(args) and fk == dict(kwargs):
        return None
    return fixed


def run_agent(
    user: str,
    generate: GenerateFn,
    *,
    history: list[str] | None = None,
    cfg: AgentConfig | None = None,
    tools: dict[str, ToolSpec] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> AgentTurn:
    """Blocking multi-step tool loop. `generate(prompt, **kwargs) -> text`.

    If `history` is a list, it is mutated in place (append one turn line) so
    multi-turn callers (e.g. chat._agent_history) accumulate context. Pass
    None to start a fresh ephemeral history.
    """
    cfg = cfg or AgentConfig()
    tools = tools or DEFAULT_TOOLS
    # Mutate caller's list when provided; only allocate when history is None.
    if history is None:
        history = []
    system = build_system(tools)
    prompt = build_prompt(user, history, system=system, tools=tools)
    transcript_parts = [prompt]
    tool_trace: list[dict] = []
    gate_events: list[dict] = []
    final = ""
    last_result: tuple[str, str] | None = None
    assisted = False
    gold_call = host_assist_call(user)  # None ⇒ user does not need a tool
    model_emitted_call = False

    for step in range(cfg.max_tool_steps + 1):
        cont = generate(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop="\nUser:",
            include_prompt=False,
        )
        if cfg.constrain_tools:
            cont = normalize_assistant_cont(cont, tools=tools)

        raw_call = parse_call(cont)
        if raw_call is not None:
            model_emitted_call = True

        # When2Call gate: if user needs no tool, never execute model tool spam
        if cfg.gate_tools and gold_call is None:
            if raw_call is not None:
                gate_events.append(
                    {
                        "kind": "block_false_call",
                        "model_tool": raw_call[0],
                        "model_args": list(raw_call[1]),
                        "step": step,
                    }
                )
            else:
                gate_events.append(
                    {
                        "kind": "pass_through_no_call",
                        "step": step,
                    }
                )
            final = _finalize_no_tool_text(user, cont)
            if on_token:
                on_token(final)
            transcript_parts.append(final)
            break

        # Product gate: if host knows the intent tool and model emitted a
        # *different* tool name (e.g. lookup for "what day is it"), rewrite
        # to gold. Not assist (does not invent a call when model is silent).
        if cfg.gate_tools and gold_call is not None and not cfg.assist_tools:
            gcall = parse_call(gold_call)
            mcall = parse_call(cont)
            if gcall is not None and mcall is not None and mcall[0] != gcall[0]:
                gate_events.append(
                    {
                        "kind": "rewrite_tool",
                        "from_tool": mcall[0],
                        "to_tool": gcall[0],
                        "step": step,
                    }
                )
                cont = gold_call
                assisted = True

        # Host gold inject: only when assist is on (never pure policy).
        if (
            cfg.host_may_inject()
            and gold_call is not None
            and (cfg.assist_tools or parse_call(cont) is None)
        ):
            cont = gold_call
            assisted = True

        if on_token and cont:
            on_token(cont)

        call = parse_call(cont)
        if call is None or step >= cfg.max_tool_steps:
            final = _finalize_no_tool_text(user, cont)
            if last_result and not last_result[1].startswith("ERROR:"):
                final = _host_phrase_answer(user, last_result[0], last_result[1])
            elif last_result and last_result[1].startswith("ERROR:") and gold_call is None:
                # False tool call on chitchat: do not surface ERROR spam to the user
                final = chitchat_reply(user) or _finalize_no_tool_text(user, cont)
            transcript_parts.append(final)
            break

        name, args, kwargs = call
        span = extract_call_span(cont) or _format_call_span(name, list(args), dict(kwargs))
        before = cont[: cont.find(span)] if span in cont else ""
        result_body = execute_call(name, args, kwargs, tools=tools)

        # Host repair: ERROR or wrong calc vs gold (never in pure policy)
        repaired = None
        if result_body.startswith("ERROR:"):
            # Pure-policy false call (no gold tool need): stop and chitchat
            if gold_call is None and not cfg.host_may_inject():
                tool_trace.append(
                    {
                        "tool": name,
                        "args": list(args),
                        "kwargs": dict(kwargs),
                        "result": result_body,
                        "repaired": False,
                    }
                )
                final = chitchat_reply(user) or "I'm not sure. Try a math question or a simple fact lookup."
                if on_token:
                    on_token(final)
                transcript_parts.append(final)
                break
            if cfg.host_may_inject():
                repaired = _try_host_repair_call(user, name, args, kwargs, result_body)
        elif name == "calc" and gold_call and cfg.host_may_inject():
            gcall = parse_call(gold_call)
            if gcall is not None:
                gn, ga, gk = gcall
                gold_body = execute_call(gn, ga, gk, tools=tools)
                if not gold_body.startswith("ERROR:") and gold_body != result_body:
                    repaired = gcall
        if repaired is not None:
            rname, rargs, rkwargs = repaired
            rspan = _format_call_span(rname, list(rargs), dict(rkwargs))
            tool_trace.append(
                {
                    "tool": name,
                    "args": list(args),
                    "kwargs": dict(kwargs),
                    "result": result_body,
                    "repaired": True,
                }
            )
            if on_token:
                on_token(
                    f"\nTool: {format_result(result_body, max_chars=cfg.max_result_chars)}\n"
                )
            name, args, kwargs = rname, rargs, rkwargs
            span = rspan
            before = ""
            result_body = execute_call(name, args, kwargs, tools=tools)
            assisted = True
            if on_token:
                on_token(f"{span}\n")

        result_tag = format_result(result_body, max_chars=cfg.max_result_chars)
        last_result = (name, result_body)
        tool_trace.append(
            {
                "tool": name,
                "args": list(args),
                "kwargs": dict(kwargs),
                "result": result_body,
            }
        )
        piece = f"{before}{span}\nTool: {result_tag}\nAssistant:"
        transcript_parts.append(piece)
        if on_token:
            on_token(f"\nTool: {result_tag}\n")

        if result_body.startswith("ERROR:"):
            final = _host_phrase_answer(user, name, result_body)
            if on_token:
                on_token(final)
            transcript_parts.append(final)
            break

        # One tool then host phrase (no multi-step model babble)
        final = _host_phrase_answer(user, name, result_body)
        if on_token:
            on_token(final)
        transcript_parts.append(final)
        break

    hist_line = f"User: {user}\nAssistant: {final}\n"
    history.append(hist_line)
    return AgentTurn(
        user=user,
        transcript="".join(transcript_parts),
        final=final,
        tool_trace=list(tool_trace),  # shallow copy; do not alias internal list
        gate_events=list(gate_events),
        model_emitted_call=bool(model_emitted_call),
        gold_needs_tool=gold_call is not None,
    )


def _CALL_RE_strip(text: str) -> str:
    from ll3m.tools import _CALL_RE

    return _CALL_RE.sub("", text)


def _host_phrase_answer(user: str, name: str, result_body: str) -> str:
    """Deterministic short answer from a tool result (bootstrap until tool SFT)."""
    if result_body.startswith("ERROR:"):
        return f"I could not complete that ({result_body})."
    if name == "calc":
        return f"The answer is {result_body}."
    if name == "time":
        return f"It is {result_body}."
    if name == "lookup":
        return f"{result_body}."
    if name == "wiki":
        return result_body if result_body.endswith(".") else result_body + "."
    if name == "note":
        return f"OK, {result_body}."
    return result_body


def _looks_degenerate(text: str) -> bool:
    """Heuristic: untrained base often loops commas / short tokens after tools."""
    t = text.strip()
    if len(t) < 2:
        return True
    if t.count(",") > max(8, len(t) // 4):
        return True
    words = t.split()
    if words and len(set(words)) <= 2 and len(words) > 6:
        return True
    if re.search(r"(?i)\b(the,\s*){4,}", t):
        return True
    # Tool-tag collapse: calcalcallcall… / calloccalloc
    if re.search(r"(?i)(cal){3,}|(call){3,}|<cal", t):
        return True
    if t.lower().count("call") >= 4 and "<" in t:
        return True
    return False


def stream_agent_text(
    user: str,
    generate_stream: Callable[..., Iterator[str]],
    generate: GenerateFn,
    *,
    history: list[str] | None = None,
    cfg: AgentConfig | None = None,
    tools: dict[str, ToolSpec] | None = None,
) -> Iterator[str]:
    """Yield user-visible pieces (answers + optional tool lines).

    Tool steps use non-stream generate for reliable parse; final answer streams.
    If `history` is a list, append the completed turn in place (multi-turn).
    """
    cfg = cfg or AgentConfig()
    tools = tools or DEFAULT_TOOLS
    if history is None:
        history = []
    system = build_system(tools)
    prompt = build_prompt(user, history, system=system, tools=tools)
    last_result: tuple[str, str] | None = None  # (tool_name, body)
    assisted = False
    gold_call = host_assist_call(user)

    for step in range(cfg.max_tool_steps + 1):
        cont = generate(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop="\nUser:",
            include_prompt=False,
        )
        if cfg.constrain_tools:
            cont = normalize_assistant_cont(cont, tools=tools)

        if cfg.gate_tools and gold_call is None:
            text = _finalize_no_tool_text(user, cont)
            if text:
                yield text
            history.append(f"User: {user}\nAssistant: {text}\n")
            return

        if cfg.gate_tools and gold_call is not None and not cfg.assist_tools:
            gcall = parse_call(gold_call)
            mcall = parse_call(cont)
            if gcall is not None and mcall is not None and mcall[0] != gcall[0]:
                cont = gold_call
                assisted = True

        if (
            cfg.host_may_inject()
            and gold_call is not None
            and (cfg.assist_tools or parse_call(cont) is None)
        ):
            cont = gold_call
            assisted = True

        call = parse_call(cont)
        if call is None or step >= cfg.max_tool_steps:
            text = _finalize_no_tool_text(user, cont)
            if last_result and not last_result[1].startswith("ERROR:"):
                text = _host_phrase_answer(user, last_result[0], last_result[1])
            elif last_result and last_result[1].startswith("ERROR:") and gold_call is None:
                text = chitchat_reply(user) or text
            if text:
                yield text
            history.append(f"User: {user}\nAssistant: {text}\n")
            return

        name, args, kwargs = call
        span = extract_call_span(cont) or _format_call_span(name, list(args), dict(kwargs))
        result_body = execute_call(name, args, kwargs, tools=tools)

        repaired = None
        if result_body.startswith("ERROR:"):
            # Pure policy false call (user needs no tool): do not surface
            # <call>/ERROR to the user stream — only the chitchat recovery.
            if gold_call is None and not cfg.host_may_inject():
                text = chitchat_reply(user) or "I'm not sure. Try a math question or a simple fact lookup."
                if text:
                    yield text
                history.append(f"User: {user}\nAssistant: {text}\n")
                return
            if cfg.host_may_inject():
                repaired = _try_host_repair_call(user, name, args, kwargs, result_body)
        elif name == "calc" and gold_call and cfg.host_may_inject():
            gcall = parse_call(gold_call)
            if gcall is not None:
                gn, ga, gk = gcall
                gold_body = execute_call(gn, ga, gk, tools=tools)
                if not gold_body.startswith("ERROR:") and gold_body != result_body:
                    repaired = gcall
        if repaired is not None:
            yield f"{span}\nTool: {format_result(result_body, max_chars=cfg.max_result_chars)}\n"
            rname, rargs, rkwargs = repaired
            span = _format_call_span(rname, list(rargs), dict(rkwargs))
            name, args, kwargs = rname, rargs, rkwargs
            result_body = execute_call(name, args, kwargs, tools=tools)
            assisted = True
        elif result_body.startswith("ERROR:"):
            result_tag = format_result(result_body, max_chars=cfg.max_result_chars)
            yield f"{span}\nTool: {result_tag}\n"
            text = _host_phrase_answer(user, name, result_body)
            yield text
            history.append(f"User: {user}\nAssistant: {text}\n")
            return

        result_tag = format_result(result_body, max_chars=cfg.max_result_chars)
        last_result = (name, result_body)
        yield f"{span}\nTool: {result_tag}\n"
        text = _host_phrase_answer(user, name, result_body)
        yield text
        history.append(f"User: {user}\nAssistant: {text}\n")
        return
