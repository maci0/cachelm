#!/usr/bin/env python3
"""Big agent + language battery against a .ll3m / .pt pack.

  uv run python scripts/eval_agent_battery.py \
    --checkpoint checkpoints/v76_base_en_tool/best.ll3m \
    --out checkpoints/v76_base_en_tool/battery_report.json

Covers: greetings/no-call, calc suite, lookup, time, wiki (optional net),
language openers, degeneracy, C vs host-gold agreement.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ll3m.agent import (
    AgentConfig,
    chitchat_reply,
    host_assist_call,
    run_agent,
)
from ll3m.chat import LL3M
from ll3m.tools import execute_call, parse_call


@dataclass
class CaseResult:
    name: str
    category: str
    user: str
    ok: bool
    detail: str
    final: str = ""
    tools: list = field(default_factory=list)
    ms: float = 0.0
    # Host-gate instrumentation (from run_agent); empty when gate off.
    gate_events: list = field(default_factory=list)
    model_emitted_call: bool = False
    gold_needs_tool: bool = False
    executed_tools: list = field(default_factory=list)
    # Dual pure scoring: ok = emission-aware (executed tools); ok_user_visible =
    # final-text-only recovery (chitchat after a false tool may still pass).
    # Freeze pure 90/101 is a historical stream-visible pin, not either scorer.
    ok_user_visible: bool = True
    detail_user_visible: str = ""


def _tools_summary(trace: list) -> list:
    out = []
    for t in trace:
        out.append(
            {
                "tool": t.get("tool"),
                "args": t.get("args"),
                "result": str(t.get("result", ""))[:120],
                "repaired": t.get("repaired"),
            }
        )
    return out


def run_case(
    m: LL3M,
    name: str,
    category: str,
    user: str,
    *,
    expect_tools: bool | None = None,
    expect_substr: list[str] | None = None,
    expect_no_substr: list[str] | None = None,
    expect_tool: str | None = None,
    expect_result_substr: list[str] | None = None,
    assist: bool = False,
    gate_tools: bool = True,
    temperature: float = 0.2,
    top_k: int = 10,
) -> CaseResult:
    """Score one case via run_agent (records host-gate interventions).

    Product scoring uses *executed* tools / final text (gate may hide false calls).
    Gate events always report what the model emitted vs what the host allowed.
    """
    m.reset_chat()
    t0 = time.perf_counter()
    cfg = AgentConfig(
        max_tokens=80,
        temperature=temperature,
        top_k=top_k,
        assist_tools=assist,
        gate_tools=gate_tools,
    )

    def gen(prompt: str, **kw) -> str:
        return m._generate_complete(prompt, **kw)

    try:
        hist: list[str] = []
        turn = run_agent(user, gen, history=hist, cfg=cfg)
        final = turn.final or ""
        full = turn.transcript or final
        gate_events = list(turn.gate_events or [])
        model_emitted = bool(turn.model_emitted_call)
        gold_needs = bool(turn.gold_needs_tool)
        executed = [str(t.get("tool", "")).lower() for t in (turn.tool_trace or [])]
        # Model-visible call names (pre-gate) from events + executed
        tools_used: list[str] = []
        for ev in gate_events:
            if ev.get("kind") == "block_false_call" and ev.get("model_tool"):
                tools_used.append(str(ev["model_tool"]).lower())
            if ev.get("kind") == "rewrite_tool" and ev.get("from_tool"):
                tools_used.append(str(ev["from_tool"]).lower())
        tools_used.extend(executed)
        # de-dupe preserve order
        seen: set[str] = set()
        tools_used = [x for x in tools_used if not (x in seen or seen.add(x))]
    except Exception as e:
        return CaseResult(
            name=name,
            category=category,
            user=user,
            ok=False,
            detail=f"exception: {e}",
            ms=(time.perf_counter() - t0) * 1000,
            ok_user_visible=False,
            detail_user_visible=f"exception: {e}",
        )
    ms = (time.perf_counter() - t0) * 1000

    # User-visible tool execution (post-gate)
    has_executed = bool(executed)
    # Degenerate babble in final/transcript
    degen = bool(
        re.search(r"(?i)(cal){3,}|(call){3,}|calloc", full)
        or full.count("<call>") >= 4
        or len(re.findall(r"(?i)\bcall\b", full)) >= 6
    )
    if (
        degen
        and expect_substr
        and any(s in final for s in expect_substr)
        and len(final) < 120
        and not re.search(r"(?i)(cal){3,}|(call){3,}|calloc", final)
    ):
        degen = False
    # Final-text-only degen (user-visible scorer ignores transcript tool spam)
    degen_final = bool(
        re.search(r"(?i)(cal){3,}|(call){3,}|calloc", final)
        or final.count("<call>") >= 4
        or len(re.findall(r"(?i)\bcall\b", final)) >= 6
    )
    errs: list[str] = []
    if degen:
        errs.append("degenerate_babble")
    # Emission-aware: expect_tools refers to *executed* tools when gated,
    # so a blocked false-call still passes no_call under product gate.
    if expect_tools is True and not has_executed:
        # Allow correct answer substr if host phrased without re-showing call
        if not (expect_substr and any(s in final for s in expect_substr)):
            errs.append("expected_tool_missing")
    if expect_tools is False and has_executed:
        errs.append(f"unexpected_tool:{executed}")
    if expect_tool and expect_tool.lower() not in executed:
        if not (expect_substr and any(s in final for s in expect_substr)):
            errs.append(f"expected_tool_name={expect_tool} got={executed}")
    if expect_substr:
        for s in expect_substr:
            if s.lower() not in final.lower() and s.lower() not in full.lower():
                errs.append(f"missing_substr:{s!r}")
    if expect_no_substr:
        for s in expect_no_substr:
            # Do not fail no_call on model emission blocked by gate (not user-visible)
            if s.lower() in ("<call>", "<call>lookup"):
                if has_executed or (not gate_tools and model_emitted):
                    if s.lower() in final.lower() or (
                        not gate_tools and s.lower() in full.lower()
                    ):
                        errs.append(f"forbidden_substr:{s!r}")
                continue
            if s.lower() in final.lower():
                errs.append(f"forbidden_substr:{s!r}")
    if expect_result_substr:
        for s in expect_result_substr:
            if s not in full and s not in final:
                errs.append(f"missing_tool_result:{s!r}")

    ok = not errs
    detail = "; ".join(errs) if errs else "pass"
    # Annotate pass detail when gate saved a no_call
    if ok and any(ev.get("kind") == "block_false_call" for ev in gate_events):
        detail = "pass;gate_blocked_false_call"
    if ok and any(ev.get("kind") == "rewrite_tool" for ev in gate_events):
        detail = (
            "pass;gate_rewrite_tool"
            if detail == "pass"
            else detail + ";gate_rewrite_tool"
        )

    # User-visible final-text scorer: ignore executed-tool / emission failures when
    # the final assistant text would still satisfy no_call / substr checks.
    # Used for pure dual metrics; freeze pure 90/101 is a separate historical pin.
    errs_uv: list[str] = []
    if degen_final:
        errs_uv.append("degenerate_babble")
    if expect_tools is True and not has_executed:
        if not (expect_substr and any(s in final for s in expect_substr)):
            errs_uv.append("expected_tool_missing")
    # Do NOT fail user-visible no_call solely because a tool executed; recovery
    # chitchat in final text can still pass if substr / forbidden checks hold.
    if expect_tool and expect_tool.lower() not in executed:
        if not (expect_substr and any(s in final for s in expect_substr)):
            errs_uv.append(f"expected_tool_name={expect_tool} got={executed}")
    if expect_substr:
        for s in expect_substr:
            if s.lower() not in final.lower():
                errs_uv.append(f"missing_substr:{s!r}")
    if expect_no_substr:
        for s in expect_no_substr:
            if s.lower() in final.lower():
                errs_uv.append(f"forbidden_substr:{s!r}")
    if expect_result_substr:
        for s in expect_result_substr:
            if s not in final and s not in full:
                # tool results may only appear in transcript; keep emission-aware
                # strictness but allow full for UV if final is empty of result
                if s not in full:
                    errs_uv.append(f"missing_tool_result:{s!r}")
    ok_uv = not errs_uv
    detail_uv = "; ".join(errs_uv) if errs_uv else "pass"

    return CaseResult(
        name=name,
        category=category,
        user=user,
        ok=ok,
        detail=detail,
        final=final[:300],
        tools=tools_used,
        ms=ms,
        gate_events=gate_events,
        model_emitted_call=model_emitted,
        gold_needs_tool=gold_needs,
        executed_tools=executed,
        ok_user_visible=ok_uv,
        detail_user_visible=detail_uv,
    )


def language_case(m: LL3M, name: str, prompt: str, *, min_alpha: float = 0.55) -> CaseResult:
    m.reset_chat()
    t0 = time.perf_counter()
    try:
        text = m.complete(
            prompt,
            max_tokens=60,
            temperature=0.7,
            top_k=30,
            include_prompt=False,
        )
    except Exception as e:
        return CaseResult(
            name=name,
            category="language",
            user=prompt[:80],
            ok=False,
            detail=f"exception: {e}",
            ms=(time.perf_counter() - t0) * 1000,
            ok_user_visible=False,
            detail_user_visible=f"exception: {e}",
        )
    ms = (time.perf_counter() - t0) * 1000
    errs = []
    if re.search(r"(?i)(cal){3,}|(call){3,}", text):
        errs.append("tool_babble_in_language")
    # printable / letter ratio
    if text:
        letters = sum(c.isalpha() or c.isspace() for c in text)
        if letters / max(len(text), 1) < min_alpha:
            errs.append(f"low_alpha_ratio={letters/len(text):.2f}")
    else:
        errs.append("empty")
    # repeated single char
    if re.search(r"(.)\1{12,}", text):
        errs.append("char_run")
    ok_lang = not errs
    detail_lang = "; ".join(errs) if errs else "pass"
    return CaseResult(
        name=name,
        category="language",
        user=prompt[:80],
        ok=ok_lang,
        detail=detail_lang,
        final=text[:300],
        ms=ms,
        ok_user_visible=ok_lang,
        detail_user_visible=detail_lang,
    )


def build_agent_cases() -> list[tuple]:
    """(name, category, user, kwargs)."""
    cases: list[tuple] = []
    # --- no-call / greetings / chitchat ---
    for u in [
        "hello",
        "hello my friend",
        "hi there",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "thanks",
        "thank you",
        "thx",
        "how are you?",
        "how are you doing?",
        "tell me a joke",
        "what is your name?",
        "who are you?",
        "ok",
        "bye",
        "see you later",
        "nice weather today",
        "I like pizza",
        "sing a song",
        "write a poem about dogs",
        "hello friend how are you today",
    ]:
        cases.append(
            (
                f"nocall:{u[:28]}",
                "no_call",
                u,
                dict(
                    expect_tools=False,
                    expect_no_substr=[
                        "<call>",
                        "ERROR:",
                        "calloc",
                        "boiling water",
                        "<call>lookup",
                    ],
                ),
            )
        )
    # --- calc (wide) ---
    calcs = [
        ("what is 5+5", "10"),
        ("what is 2+2", "4"),
        ("what is 3+4", "7"),
        ("what is 10-3", "7"),
        ("what is 6 times 7", "42"),
        ("what is 8*9", "72"),
        ("calculate 12 + 8", "20"),
        ("what is 100/4", "25"),
        ("what's 15 plus 5", "20"),
        ("What is 9 minus 2?", "7"),
        ("compute 7 x 8", "56"),
        ("what is 0+0", "0"),
        ("what is 1+1", "2"),
        ("what is 99+1", "100"),
        ("what is 50-25", "25"),
        ("what is 11+22", "33"),
        ("what is 7+8", "15"),
        ("what is 4*5", "20"),
        ("what is 9*9", "81"),
        ("what is 16/2", "8"),
        ("what is 3 times 3", "9"),
        ("what is 12 times 12", "144"),
        ("calculate 1 + 2", "3"),
        ("what's 20 minus 5", "15"),
        ("what is 2+3?", "5"),
        ("What is 100 + 1", "101"),
        ("compute 6 + 6", "12"),
        ("what is 8 plus 8", "16"),
        ("what is 10 times 10", "100"),
        ("what is 81/9", "9"),
    ]
    for u, ans in calcs:
        cases.append(
            (
                f"calc:{u}",
                "calc",
                u,
                dict(
                    expect_tools=True,
                    expect_tool="calc",
                    expect_substr=[ans],
                    expect_no_substr=["calloc", "ERROR: usage"],
                ),
            )
        )
    # --- lookup (closed KB) ---
    lookups = [
        ("what is the capital of france", "Paris"),
        ("capital of japan", "Tokyo"),
        ("capital of germany", "Berlin"),
        ("capital of italy", "Rome"),
        ("capital of spain", "Madrid"),
        ("capital of canada", "Ottawa"),
        ("capital of australia", "Canberra"),
        ("capital of india", "New Delhi"),
        ("capital of china", "Beijing"),
        ("capital of brazil", "Brasilia"),
        ("capital of egypt", "Cairo"),
        ("capital of russia", "Moscow"),
        ("what is pi", "3.14"),
        ("water formula", "H2O"),
        ("boiling water c", "100"),
        ("freezing water c", "0"),
        ("speed of light", "299792458"),
        ("largest ocean", "Pacific"),
        ("days in week", "7"),
        ("hours in day", "24"),
    ]
    for u, ans in lookups:
        cases.append(
            (
                f"lookup:{u}",
                "lookup",
                u,
                dict(
                    expect_tools=True,
                    expect_substr=[ans],
                    expect_no_substr=["calloc"],
                ),
            )
        )
    # --- time ---
    for u in [
        "what time is it?",
        "current time UTC",
        "what is the current time",
        "what day is it",
        "date today",
    ]:
        cases.append(
            (
                f"time:{u[:24]}",
                "time",
                u,
                dict(expect_tools=True, expect_tool="time", expect_no_substr=["calloc"]),
            )
        )
    # --- adversarial ---
    for name, u, kw in [
        (
            "adv:glued_spoken",
            "what is 5+5?",
            dict(expect_substr=["10"], expect_no_substr=["50", "calloc"]),
        ),
        (
            "adv:glued_34",
            "what is 3+4?",
            dict(expect_substr=["7"], expect_no_substr=["34", "calloc"]),
        ),
        (
            "adv:no_wiki_hello",
            "hello friend how are you today",
            dict(expect_tools=False, expect_no_substr=["<call>lookup", "wiki", "<call>"]),
        ),
        (
            "adv:no_tool_story",
            "once upon a time there was a cat",
            dict(expect_tools=False, expect_no_substr=["calloc", "ERROR:", "<call>"]),
        ),
        (
            "adv:no_random_boiling",
            "hello my friend",
            dict(expect_tools=False, expect_no_substr=["boiling", "<call>lookup", "<call>"]),
        ),
        (
            "adv:math_words",
            "what is 6 times 7",
            dict(expect_substr=["42"], expect_no_substr=["calloc"]),
        ),
    ]:
        cases.append((name, "adversarial", u, kw))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v76_base_en_tool/best.ll3m",
    )
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--python", action="store_true", help="force torch path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-assist",
        action="store_true",
        help="no host gold inject (assist_tools=False). When2Call gate stays on unless --no-gate.",
    )
    ap.add_argument(
        "--no-gate",
        action="store_true",
        help="also disable When2Call gate (full pure model policy including no_call)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="agent sampling temperature (default: 0.0 pure-policy / 0.2 gated)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="agent top_k (default: 1 pure-policy / 10 gated)",
    )
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"missing {ck}")

    # Default: assist off, When2Call gate on (host blocks tools on chitchat only).
    # --no-assist: no gold inject. --no-gate: also disable When2Call (strict pure no_call).
    # Product pure-tool claim: --no-assist (model must emit calc/lookup; gate for no_call).
    assist = False
    if args.no_assist:
        gate_tools = not args.no_gate
        mode = "no_assist_no_gate" if args.no_gate else "no_assist_gate_on"
    else:
        gate_tools = True
        mode = "gate_on_assist_off"
    # Greedy default when measuring pure tool emission (--no-assist).
    if args.temperature is not None:
        sample_temp = float(args.temperature)
    else:
        sample_temp = 0.0 if args.no_assist else 0.2
    if args.top_k is not None:
        sample_top_k = int(args.top_k)
    else:
        sample_top_k = 1 if args.no_assist else 10
    print(
        f"=== battery {ck} engine={'python' if args.python else 'c'} "
        f"mode={mode} assist={assist} gate={gate_tools} "
        f"temp={sample_temp} top_k={sample_top_k} ==="
    )
    m = LL3M(ck, python_infer=args.python, seed=args.seed)

    results: list[CaseResult] = []
    t0 = time.perf_counter()

    # Host unit checks (no model)
    for u, want in [
        ("hello", True),
        ("what is 5+5", False),
        ("capital of france", False),
    ]:
        is_chat = host_assist_call(u) is None
        ok = is_chat if want else not is_chat
        results.append(
            CaseResult(
                name=f"host_gate:{u}",
                category="host",
                user=u,
                ok=ok,
                detail="pass" if ok else f"assist={host_assist_call(u)!r}",
                ok_user_visible=ok,
                detail_user_visible="pass" if ok else f"assist={host_assist_call(u)!r}",
            )
        )

    # Agent battery
    for name, cat, user, kw in build_agent_cases():
        print(f"  … {name}", flush=True)
        r = run_case(
            m,
            name,
            cat,
            user,
            assist=assist,
            gate_tools=gate_tools,
            temperature=sample_temp,
            top_k=sample_top_k,
            **kw,
        )
        results.append(r)
        status = "OK" if r.ok else "FAIL"
        print(f"    {status} {r.detail[:80]}  [{r.ms:.0f}ms]  final={r.final[:60]!r}")

    # Language openers
    lang_prompts = [
        ("lang:edu", "The purpose of education is"),
        ("lang:cities", "In modern cities,"),
        ("lang:gravity", "User: Explain gravity in simple terms.\nAssistant:"),
        ("lang:water", "User: What is water made of?\nAssistant:"),
        ("lang:once", "Once upon a time"),
        ("lang:science", "Scientists have found that"),
        ("lang:history", "In the early twentieth century,"),
        ("lang:nature", "Deep in the forest,"),
        ("lang:tech", "A computer program is"),
        ("lang:health", "To stay healthy, people should"),
        ("lang:space", "The planet Mars is"),
        ("lang:ocean", "The ocean is important because"),
    ]
    for name, prompt in lang_prompts:
        print(f"  … {name}", flush=True)
        r = language_case(m, name, prompt)
        results.append(r)
        print(f"    {'OK' if r.ok else 'FAIL'} {r.detail}  final={r.final[:60]!r}")

    # Encode consistency: pack BPE vs product tokenizer (data/tokenizer.json),
    # falling back to data/base_en only if the product tokenizer is missing.
    try:
        from ll3m.c_engine import CEngine, resolve_tokenizer_path
        from ll3m.tokenizer import BPETokenizer

        if ck.suffix == ".ll3m" and not args.python:
            eng = CEngine(ck)
            tok_path = resolve_tokenizer_path(ck)
            sample = "User: What is 6 times 7?\nAssistant:"
            ids_e = eng.encode(sample)
            ok = True
            detail = f"tok={tok_path}"
            for ref_name in (
                "data/tokenizer.json",
                "data/base_en/tokenizer.json",
                "data/base_en_dense/tokenizer.json",
            ):
                ref = Path(ref_name)
                if not ref.is_file():
                    continue
                ids_b = BPETokenizer.load(ref).encode(sample)
                ok = ids_e == ids_b
                detail = f"pack_matches_{ref.name}" if ok else f"pack_ne_{ref.name}"
                if ok:
                    break
            results.append(
                CaseResult(
                    name="tok:pack_encode",
                    category="tokenizer",
                    user=sample[:40],
                    ok=ok,
                    detail=detail,
                    ok_user_visible=ok,
                    detail_user_visible=detail,
                )
            )
    except Exception as e:
        results.append(
            CaseResult(
                name="tok:pack_encode",
                category="tokenizer",
                user="",
                ok=False,
                detail=str(e),
                ok_user_visible=False,
                detail_user_visible=str(e),
            )
        )

    elapsed = time.perf_counter() - t0
    by_cat: dict[str, dict] = {}
    for r in results:
        d = by_cat.setdefault(r.category, {"n": 0, "pass": 0, "fail": 0})
        d["n"] += 1
        if r.ok:
            d["pass"] += 1
        else:
            d["fail"] += 1

    n_ok = sum(1 for r in results if r.ok)

    # Host-gate aggregate (agent cases only; host/lang/tok may be empty)
    n_block = 0
    n_rewrite = 0
    n_model_call = 0
    n_gold_need = 0
    n_false_call_emitted = 0  # model called when gold needs no tool
    n_false_call_blocked = 0
    n_true_call_emitted = 0
    n_silent_when_needed = 0  # gold needs tool, model silent
    for r in results:
        if r.category in ("host", "language", "tokenizer"):
            continue
        if r.model_emitted_call:
            n_model_call += 1
        if r.gold_needs_tool:
            n_gold_need += 1
            if r.model_emitted_call:
                n_true_call_emitted += 1
            else:
                n_silent_when_needed += 1
        else:
            if r.model_emitted_call:
                n_false_call_emitted += 1
        for ev in r.gate_events or []:
            k = ev.get("kind")
            if k == "block_false_call":
                n_block += 1
                n_false_call_blocked += 1
            elif k == "rewrite_tool":
                n_rewrite += 1

    gate_stats = {
        "gate_tools": gate_tools,
        "assist_tools": assist,
        "n_agent_cases": sum(
            1 for r in results if r.category not in ("host", "language", "tokenizer")
        ),
        "n_model_emitted_call": n_model_call,
        "n_gold_needs_tool": n_gold_need,
        "n_false_call_emitted": n_false_call_emitted,
        "n_false_call_blocked": n_false_call_blocked,
        "n_tool_rewrite": n_rewrite,
        "n_true_call_emitted": n_true_call_emitted,
        "n_silent_when_tool_needed": n_silent_when_needed,
        "note": (
            "Product pass/fail uses executed tools + final text. "
            "block_false_call means host suppressed a model tool on no-tool intent. "
            "rewrite_tool means host rewrote wrong tool name to gold intent."
        ),
    }

    # Dual pure scoring aggregates (always emitted; product gate-on UV ≈ emission).
    n_ok_uv = sum(1 for r in results if r.ok_user_visible)
    by_cat_uv: dict[str, dict[str, int]] = {}
    for r in results:
        d = by_cat_uv.setdefault(r.category, {"n": 0, "pass": 0})
        d["n"] += 1
        if r.ok_user_visible:
            d["pass"] += 1

    scoring_note = (
        "passed / pass_rate / by_category = emission-aware (executed tools + "
        "existing rules; false tool on no_call fails). "
        "passed_user_visible / pass_rate_user_visible / by_category_user_visible = "
        "final-text-only recovery (chitchat after a false tool may still pass). "
        "Freeze pure 90/101 under paper_freeze_v072 is a historical stream-visible "
        "pin and is not regenerated by either scorer."
    )

    report = {
        "checkpoint": str(ck),
        "engine": "python" if args.python else "c",
        "seed": args.seed,
        "mode": mode,
        "assist_tools": assist,
        "gate_tools": gate_tools,
        "temperature": sample_temp,
        "top_k": sample_top_k,
        "elapsed_s": elapsed,
        "total": len(results),
        "passed": n_ok,
        "failed": len(results) - n_ok,
        "pass_rate": n_ok / max(len(results), 1),
        "passed_user_visible": n_ok_uv,
        "pass_rate_user_visible": n_ok_uv / max(len(results), 1),
        "by_category": by_cat,
        "by_category_user_visible": by_cat_uv,
        "scoring_note": scoring_note,
        "gate_stats": gate_stats,
        "failures": [asdict(r) for r in results if not r.ok],
        "all": [asdict(r) for r in results],
    }

    out = Path(args.out or (ck.parent / "battery_report.json"))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=== summary ===")
    print(
        f"passed {n_ok}/{len(results)}  ({100*report['pass_rate']:.1f}%)  "
        f"user_visible {n_ok_uv}/{len(results)}  "
        f"({100*report['pass_rate_user_visible']:.1f}%)  {elapsed:.1f}s"
    )
    for cat, d in sorted(by_cat.items()):
        du = by_cat_uv.get(cat, {"pass": 0, "n": 0})
        print(
            f"  {cat:12} emission {d['pass']}/{d['n']}  "
            f"user_visible {du['pass']}/{du['n']}"
        )
    print("=== gate_stats ===")
    print(
        f"  model_calls={n_model_call}  false_call_emitted={n_false_call_emitted}  "
        f"blocked={n_false_call_blocked}  rewrite={n_rewrite}  "
        f"silent_when_needed={n_silent_when_needed}"
    )
    if report["failures"]:
        print("=== failures (emission-aware) ===")
        for r in report["failures"]:
            print(f"  [{r['category']}] {r['name']}: {r['detail']}")
            print(f"      final={r['final'][:100]!r}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
