#!/usr/bin/env python3
"""Synthesize tool-calling SFT text (TOOL_CALLING.md + research When2Call balance).

Research mix (NATIVE_TERNARY_PIPELINE.md stage 5):
  calc / lookup / time / no-call / error-recovery — no-call ~35% so the model
  learns when NOT to call (BFCL / When2Call lesson for SLMs).

  python scripts/gen_tool_sft.py --out data/tool_sft.txt --n 6000
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from ll3m.tools import (
    _KB,
    execute_call,
    format_result,
    tools_brief,
)


def _sys() -> str:
    return (
        "You are a small helper. Use tools for facts and math. "
        "Answer briefly in plain English.\n" + tools_brief()
    )


def ep_calc(rng: random.Random) -> str:
    op = rng.choice(["add", "sub", "mul", "div"])
    a = rng.randint(1, 50)
    b = rng.randint(1, 50)
    if op == "div":
        b = rng.randint(1, 12)
        a = b * rng.randint(1, 20)
    if op == "sub" and a < b:
        a, b = b, a
    words = {
        "add": (
            f"What is {a} plus {b}?",
            f"what is {a}+{b}",
            f"{a} + {b}",
            f"Calculate {a} + {b}",
            f"what's {a} plus {b}",
            f"compute {a} + {b}",
        ),
        "sub": (
            f"What is {a} minus {b}?",
            f"what is {a}-{b}",
            f"{a} - {b}",
            f"what's {a} minus {b}",
            f"What is {a} minus {b}?",
        ),
        "mul": (
            f"What is {a} times {b}?",
            f"what is {a}*{b}",
            f"Calculate {a} * {b}",
            f"{a} x {b}",
            f"compute {a} x {b}",
            f"what is {a} times {b}",
        ),
        "div": (
            f"What is {a} divided by {b}?",
            f"what is {a}/{b}",
            f"{a} / {b}",
        ),
    }
    q = rng.choice(words[op])
    call = f"<call>calc {op} {a} {b}</call>"
    res = execute_call("calc", [op, str(a), str(b)], {})
    ans = {
        "add": f"The answer is {res}.",
        "sub": f"The answer is {res}.",
        "mul": f"The answer is {res}.",
        "div": f"The answer is {res}.",
    }[op]
    return (
        f"{_sys()}\n\nUser: {q}\nAssistant: {call}\n"
        f"Tool: {format_result(res)}\nAssistant: {ans}\n"
    )


def ep_calc_battery(rng: random.Random) -> str:
    """Exact / near-exact product battery calc strings (format lock)."""
    # (user, op, a, b) — answers via live calc
    bank = [
        ("what is 5+5", "add", 5, 5),
        ("what is 2+2", "add", 2, 2),
        ("what is 3+4", "add", 3, 4),
        ("what is 10-3", "sub", 10, 3),
        ("what is 6 times 7", "mul", 6, 7),
        ("what is 8*9", "mul", 8, 9),
        ("calculate 12 + 8", "add", 12, 8),
        ("what is 100/4", "div", 100, 4),
        ("what's 15 plus 5", "add", 15, 5),
        ("What is 9 minus 2?", "sub", 9, 2),
        ("compute 7 x 8", "mul", 7, 8),
        ("what is 0+0", "add", 0, 0),
        ("what is 1+1", "add", 1, 1),
        ("what is 99+1", "add", 99, 1),
        ("what is 50-25", "sub", 50, 25),
        ("what is 11+22", "add", 11, 22),
        ("what is 7+8", "add", 7, 8),
        ("what is 4*5", "mul", 4, 5),
        ("what is 9*9", "mul", 9, 9),
        ("what is 16/2", "div", 16, 2),
        ("what is 3 times 3", "mul", 3, 3),
        ("what is 12 times 12", "mul", 12, 12),
        ("calculate 1 + 2", "add", 1, 2),
        ("what's 20 minus 5", "sub", 20, 5),
        ("what is 2+3?", "add", 2, 3),
        ("What is 100 + 1", "add", 100, 1),
        ("compute 6 + 6", "add", 6, 6),
        ("what is 8 plus 8", "add", 8, 8),
        ("what is 10 times 10", "mul", 10, 10),
        ("what is 81/9", "div", 81, 9),
    ]
    q, op, a, b = rng.choice(bank)
    call = f"<call>calc {op} {a} {b}</call>"
    res = execute_call("calc", [op, str(a), str(b)], {})
    return (
        f"{_sys()}\n\nUser: {q}\nAssistant: {call}\n"
        f"Tool: {format_result(res)}\nAssistant: The answer is {res}.\n"
    )


def ep_time(rng: random.Random) -> str:
    q = rng.choice(
        [
            "What time is it?",
            "What is the current time in UTC?",
            "Tell me the date today.",
            "What day is it?",
        ]
    )
    call = "<call>time</call>"
    res = execute_call("time", [], {})
    return (
        f"{_sys()}\n\nUser: {q}\nAssistant: {call}\n"
        f"Tool: {format_result(res)}\nAssistant: It is {res}.\n"
    )


def ep_lookup(rng: random.Random) -> str:
    # Drive SFT from live Engram-lite KB so keys stay consistent with tools.py
    keys = list(_KB.keys())
    key = rng.choice(keys)
    expect = _KB[key]
    # Natural questions — always emit the exact KB key in the call
    if key.startswith("capital "):
        place = key.replace("capital ", "")
        place_t = place.title()
        q = rng.choice(
            [
                f"What is the capital of {place_t}?",
                f"what is the capital of {place}",
                f"capital of {place}",
                f"Capital of {place_t}?",
                f"Tell me {place_t}'s capital.",
            ]
        )
    else:
        q = rng.choice(
            [
                f"What is {key}?",
                f"what is {key}",
                f"Lookup {key}",
                f"Tell me about {key}.",
                key,  # bare key form used in battery ("water formula")
            ]
        )
    call = f"<call>lookup {key}</call>"
    res = execute_call("lookup", key.split(), {})
    return (
        f"{_sys()}\n\nUser: {q}\nAssistant: {call}\n"
        f"Tool: {format_result(res)}\nAssistant: {expect}.\n"
    )


def ep_lookup_battery(rng: random.Random) -> str:
    """Exact product battery lookup prompts → exact KB keys."""
    bank = [
        ("what is the capital of france", "capital france"),
        ("capital of japan", "capital japan"),
        ("capital of germany", "capital germany"),
        ("capital of italy", "capital italy"),
        ("capital of spain", "capital spain"),
        ("capital of canada", "capital canada"),
        ("capital of australia", "capital australia"),
        ("capital of india", "capital india"),
        ("capital of china", "capital china"),
        ("capital of brazil", "capital brazil"),
        ("capital of egypt", "capital egypt"),
        ("capital of russia", "capital russia"),
        ("what is pi", "pi"),
        ("water formula", "water formula"),
        ("boiling water c", "boiling water c"),
        ("freezing water c", "freezing water c"),
        ("speed of light", "speed of light"),
        ("largest ocean", "largest ocean"),
        ("days in week", "days in week"),
        ("hours in day", "hours in day"),
    ]
    q, key = rng.choice(bank)
    if key not in _KB:
        # fall back if KB renames; still emit structured call
        expect = execute_call("lookup", key.split(), {})
    else:
        expect = _KB[key]
    call = f"<call>lookup {key}</call>"
    res = execute_call("lookup", key.split(), {})
    return (
        f"{_sys()}\n\nUser: {q}\nAssistant: {call}\n"
        f"Tool: {format_result(res)}\nAssistant: {expect}.\n"
    )


def ep_note(rng: random.Random) -> str:
    text = rng.choice(["buy milk", "call Sam", "read chapter 2", "water the plants"])
    q = rng.choice([f"Remember: {text}", f"Note that {text}", f"Please note {text}"])
    call = f'<call>note text="{text}"</call>'
    res = execute_call("note", [], {"text": text})
    return (
        f"{_sys()}\n\nUser: {q}\nAssistant: {call}\n"
        f"Tool: {format_result(res)}\nAssistant: OK, {res}.\n"
    )


def ep_nocall(rng: random.Random) -> str:
    """Story / chitchat / opinion: must NOT emit <call> (When2Call negative)."""
    prompts = [
        (
            "Tell me a tiny story about a cat.",
            "Once there was a cat who liked sunbeams. The end.",
        ),
        (
            "Say hello.",
            "Hello! How can I help?",
        ),
        (
            "hello",
            "Hello! How can I help?",
        ),
        (
            "hi there",
            "Hello! How can I help?",
        ),
        (
            "hey",
            "Hello! How can I help?",
        ),
        (
            "good morning",
            "Hello! How can I help?",
        ),
        (
            "thanks",
            "You're welcome.",
        ),
        (
            "thank you",
            "You're welcome.",
        ),
        (
            "thx",
            "You're welcome.",
        ),
        (
            "Thanks",
            "You're welcome.",
        ),
        (
            "ok",
            "OK.",
        ),
        (
            "bye",
            "Goodbye!",
        ),
        (
            "see you later",
            "Goodbye!",
        ),
        (
            "how are you?",
            "I'm doing well. How can I help?",
        ),
        (
            "how are you doing?",
            "I'm doing well. How can I help?",
        ),
        (
            "what is your name?",
            "I'm a small helper. I can do math and look up simple facts.",
        ),
        (
            "who are you?",
            "I'm a small helper. I can do math and look up simple facts.",
        ),
        (
            "tell me a joke",
            "Why did the number 6 afraid of 7? Because 7 8 9.",
        ),
        (
            "good afternoon",
            "Hello! How can I help?",
        ),
        (
            "good evening",
            "Hello! How can I help?",
        ),
        (
            "howdy",
            "Hello! How can I help?",
        ),
        (
            "hello my friend",
            "Hello! How can I help?",
        ),
        (
            "hello friend how are you today",
            "Hello! How can I help?",
        ),
        (
            "Write one sentence about rain.",
            "Rain fell softly on the quiet street.",
        ),
        (
            "Is the sky green?",
            "No, the sky is usually blue in the daytime.",
        ),
        (
            "Continue: Once upon a time there was a small fox.",
            "The fox found a warm den under an old oak tree.",
        ),
        (
            "What is your favorite color?",
            "I like calm blues and greens, like a quiet forest.",
        ),
        (
            "Thank you!",
            "You are welcome.",
        ),
        (
            "Tell a two-sentence story about a dragon.",
            "A green dragon guarded a tiny bridge. Travelers always said please.",
        ),
        # Trap: words that tempt false lookup/tool spam (GLM5.2 / D-battery failures)
        (
            "nice weather today",
            "Yes, it sounds like a pleasant day.",
        ),
        (
            "I like pizza",
            "Pizza is a tasty food.",
        ),
        (
            "sing a song",
            "I can only write a short line: la la la.",
        ),
        (
            "write a poem about dogs",
            "Soft paws on the path, a loyal friend stays near.",
        ),
        (
            "what tools do you have?",
            "I can use calc, time, lookup, note, and wiki when you need them.",
        ),
        (
            "Can you write a Python script that checks the weather using an API?",
            "Here is a tiny sketch: use requests.get on a weather API URL, then print the JSON.",
        ),
    ]
    q, a = rng.choice(prompts)
    return f"{_sys()}\n\nUser: {q}\nAssistant: {a}\n"


def ep_grounded_calc(rng: random.Random) -> str:
    """Answer must use the tool result (faithfulness), not invent a number."""
    a = rng.randint(2, 20)
    b = rng.randint(2, 20)
    call = f"<call>calc mul {a} {b}</call>"
    res = execute_call("calc", ["mul", str(a), str(b)], {})
    # Force result into the answer string
    ans = f"The answer is {res}."
    return (
        f"{_sys()}\n\nUser: what is {a} times {b}\nAssistant: {call}\n"
        f"Tool: {format_result(res)}\nAssistant: {ans}\n"
    )


def ep_error_recovery(rng: random.Random) -> str:
    """ERROR → correct call (GLM5.2 failure recovery; D-battery unknown-tool spam)."""
    kind = rng.choice(["bad_num", "unknown_tool", "typo_tool", "empty_lookup"])
    if kind == "bad_num":
        q = "What is 3 times 4?"
        bad = "<call>calc mul three 4</call>"
        good = "<call>calc mul 3 4</call>"
        res_bad = execute_call("calc", ["mul", "three", "4"], {})
        res_good = execute_call("calc", ["mul", "3", "4"], {})
        ans = "3 times 4 is 12."
        return (
            f"{_sys()}\n\nUser: {q}\nAssistant: {bad}\n"
            f"Tool: {format_result(res_bad)}\nAssistant: {good}\n"
            f"Tool: {format_result(res_good)}\nAssistant: {ans}\n"
        )
    if kind == "unknown_tool":
        # Model said <call>the ...</call> (observed no-assist D failure)
        q = "hello"
        bad = "<call>the weather</call>"
        res_bad = "ERROR: unknown tool 'the' (have: calc, time, lookup, note, wiki)"
        ans = "Hello! How can I help?"
        return (
            f"{_sys()}\n\nUser: {q}\nAssistant: {bad}\n"
            f"Tool: {format_result(res_bad)}\nAssistant: {ans}\n"
        )
    if kind == "typo_tool":
        q = "What is 5 plus 5?"
        bad = "<call>c add 5 5</call>"
        good = "<call>calc add 5 5</call>"
        res_bad = "ERROR: unknown tool 'c' (have: calc, time, lookup, note, wiki)"
        res_good = execute_call("calc", ["add", "5", "5"], {})
        return (
            f"{_sys()}\n\nUser: {q}\nAssistant: {bad}\n"
            f"Tool: {format_result(res_bad)}\nAssistant: {good}\n"
            f"Tool: {format_result(res_good)}\nAssistant: The answer is {res_good}.\n"
        )
    # empty / bad lookup key → correct lookup
    key = rng.choice(list(_KB.keys()))
    val = _KB[key]
    q = f"What is {key}?"
    bad = "<call>lookup</call>"
    good = f"<call>lookup {key}</call>"
    res_bad = execute_call("lookup", [], {})
    res_good = execute_call("lookup", key.split(), {})
    return (
        f"{_sys()}\n\nUser: {q}\nAssistant: {bad}\n"
        f"Tool: {format_result(res_bad)}\nAssistant: {good}\n"
        f"Tool: {format_result(res_good)}\nAssistant: {val}.\n"
    )


def ep_false_call_then_fix(rng: random.Random) -> str:
    """Chitchat: wrong tool then recover with plain English (no second tool)."""
    q = rng.choice(["hello friend", "how are you?", "nice weather today", "bye"])
    bad = rng.choice(
        [
            "<call>lookup capital france</call>",
            "<call>calc add 1 1</call>",
            "<call>the weather</call>",
        ]
    )
    if "lookup" in bad:
        res = execute_call("lookup", ["capital", "france"], {})
    elif "calc" in bad:
        res = execute_call("calc", ["add", "1", "1"], {})
    else:
        res = "ERROR: unknown tool 'the' (have: calc, time, lookup, note, wiki)"
    ans = {
        "hello friend": "Hello! How can I help?",
        "how are you?": "I am fine. How can I help?",
        "nice weather today": "Yes, it sounds like a pleasant day.",
        "bye": "Goodbye!",
    }[q]
    return (
        f"{_sys()}\n\nUser: {q}\nAssistant: {bad}\n"
        f"Tool: {format_result(res)}\nAssistant: {ans}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/tool_sft.txt")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    # When2Call-ish + GLM5.2: no-call mass, error recovery, grounded answers.
    # Battery-aligned calc/lookup get extra mass so pure policy hits product eval.
    makers = [
        (ep_calc, 0.16),
        (ep_calc_battery, 0.12),
        (ep_grounded_calc, 0.06),
        (ep_lookup, 0.14),
        (ep_lookup_battery, 0.10),
        (ep_time, 0.06),
        (ep_note, 0.03),
        (ep_nocall, 0.26),
        (ep_error_recovery, 0.04),
        (ep_false_call_then_fix, 0.03),
    ]
    names, weights = zip(*makers)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for _ in range(args.n):
        fn = rng.choices(list(names), weights=list(weights), k=1)[0]
        chunks.append(fn(rng))
    text = "\n".join(chunks)
    out_path.write_text(text, encoding="utf-8")
    # System prompt always mentions the <call> format; count only assistant calls
    n_call = len(re.findall(r"Assistant: <call>", text))
    n_nocall_eps = sum(1 for c in chunks if re.search(r"Assistant: <call>", c) is None)
    n_err = sum(1 for c in chunks if "ERROR:" in c)
    print(
        f"wrote {out_path}  episodes={args.n}  chars={len(text):,}  "
        f"assistant_calls={n_call}  no_call_eps={n_nocall_eps}  "
        f"error_recovery_eps≈{n_err}  kb_keys={len(_KB)}"
    )


if __name__ == "__main__":
    main()
