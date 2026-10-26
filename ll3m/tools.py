"""Closed tool registry for ll3m agent (host-side execution only).

Wire format (docs/TOOL_CALLING.md):
  Assistant: <call>name arg1 arg2 key=val</call>
  Tool: <result>...</result>
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


# Tags must match training / agent parse exactly.
CALL_OPEN = "<call>"
CALL_CLOSE = "</call>"
RESULT_OPEN = "<result>"
RESULT_CLOSE = "</result>"

_CALL_RE = re.compile(
    r"<call>\s*([a-zA-Z_][\w]*)\s*(.*?)\s*</call>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    doc: str
    run: Callable[..., str]


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _word_bounded(hay: str, needle: str) -> bool:
    """True if needle appears in hay with non-word boundaries (pi not in pizza)."""
    if not needle or not hay:
        return False
    if hay == needle:
        return True
    return re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", hay) is not None


# Closed KB for lookup (no network). Engram-lite: facts outside ternary weights
# (DeepSeek Engram thesis). kb_facts.json may be:
#   v1 flat {key: value}
#   v2 {version:2, entries:[{id,value,aliases}], flat:{...}}
def _load_kb() -> dict[str, str]:
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "kb_facts.json"
    if not path.is_file():
        return {
            "capital france": "Paris",
            "water formula": "H2O",
            "pi": "3.1415926535",
            "days in week": "7",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"kb_facts.json empty or invalid: {path}")
    # v2: prefer precomputed flat, else expand entries
    if data.get("version") == 2 or "entries" in data:
        flat = data.get("flat")
        if isinstance(flat, dict) and flat:
            return {_norm_key(str(k)): str(v) for k, v in flat.items()}
        out: dict[str, str] = {}
        for ent in data.get("entries") or []:
            if not isinstance(ent, dict):
                continue
            val = str(ent.get("value", "")).strip()
            eid = _norm_key(str(ent.get("id", "")))
            if eid and val:
                out[eid] = val
            for a in ent.get("aliases") or []:
                ak = _norm_key(str(a))
                if ak and val and ak not in out:
                    out[ak] = val
        if out:
            return out
    # v1 flat map (skip metadata keys)
    skip = {"version", "note", "n_entries", "n_keys", "entries", "flat"}
    return {
        _norm_key(str(k)): str(v)
        for k, v in data.items()
        if k not in skip and not isinstance(v, (dict, list))
    }


_KB: dict[str, str] = _load_kb()


def get_kb() -> dict[str, str]:
    """Live process-global lookup table (always the current mapping).

    Prefer this over importing ``_KB`` by name: ``from ll3m.tools import _KB``
    binds a reference that goes stale if the dict object is replaced.
    """
    return _KB


def set_kb(flat: dict[str, str] | None = None) -> dict[str, str]:
    """Replace or restore the process-global lookup table (tests / ablations).

    Mutates the existing dict in place so imported aliases (and any holders of
    the original object) stay live. Pass None to reload from ll3m/kb_facts.json.
    Pass {} for empty KB (off).
    """
    if flat is None:
        new = _load_kb()
    else:
        new = {_norm_key(str(k)): str(v) for k, v in flat.items()}
    _KB.clear()
    _KB.update(new)
    return _KB


def kb_size() -> int:
    return len(_KB)



def _format_calc_num(v: float | int) -> str:
    """Format a calc result; never raise on nan/inf."""
    if isinstance(v, float):
        if math.isnan(v):
            return "ERROR: not a number"
        if math.isinf(v):
            return "ERROR: overflow"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
    return str(v)


def tool_calc(*args: str, **kwargs: str) -> str:
    """calc op a b  |  calc expr=..."""
    if "expr" in kwargs:
        expr = kwargs["expr"]
        # Safe eval: numbers and + - * / ( ) . only
        if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
            return "ERROR: unsafe expr"
        try:
            val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — filtered charset
            if isinstance(val, (int, float)):
                return _format_calc_num(float(val))
            return "ERROR: bad result type"
        except Exception as e:
            return f"ERROR: {e}"
    if len(args) < 3:
        return "ERROR: usage calc op a b"
    op, a_s, b_s = args[0], args[1], args[2]
    try:
        a, b = float(a_s), float(b_s)
    except ValueError:
        return "ERROR: need numbers"
    if not math.isfinite(a) or not math.isfinite(b):
        return "ERROR: need finite numbers"
    # Bound pow to avoid CPU DoS / overflow (agent host is local but still)
    if op == "pow":
        if abs(b) > 100 or (abs(a) > 1e6 and abs(b) > 10):
            return "ERROR: pow exponent too large"
    if op == "div":
        if b == 0:
            return "ERROR: division by zero"
        v = a / b
    elif op == "mod":
        if b == 0:
            return "ERROR: division by zero"
        v = a % b
    elif op == "add":
        v = a + b
    elif op == "sub":
        v = a - b
    elif op == "mul":
        v = a * b
    elif op == "pow":
        try:
            v = a**b
        except OverflowError:
            return "ERROR: overflow"
    else:
        return f"ERROR: unknown op {op} (add|sub|mul|div|pow|mod)"
    return _format_calc_num(v)


def tool_time(*_args: str, **_kwargs: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def tool_lookup(*args: str, **kwargs: str) -> str:
    q = kwargs.get("q") or " ".join(args)
    key = _norm_key(q)
    key = re.sub(r"^(what\s+is|what's|whats|tell\s+me|define)\s+", "", key).strip(" ?.")
    if key in _KB:
        return _KB[key]
    # Fuzzy match without substring traps (pi ⊂ pizza) and without short
    # queries latching onto unrelated keys that merely contain the word
    # (e.g. "cpu" → "cpu full form", never "cpu cache" / "cpu l3" / "… CPU …").
    # Rank tuple: lower is better.
    #   tier 0: stored key is a word-bounded phrase inside the query (longest key)
    #   tier 1: query is a clean prefix of stored key; prefer definitional suffixes
    #   tier 2: multi-word query is word-bounded inside a stored key (shortest)
    _DEF_SUFFIX = {
        "full form": 0,
        "full name": 0,
        "means": 1,
        "stands for": 1,
        "definition": 1,
        "formula": 2,
        "capital": 2,
    }
    best: tuple[int, int, int, str] | None = None  # tier, sub, len_metric → then value via map
    best_v = ""
    multi = " " in key
    for k, v in _KB.items():
        if len(k) < 2:
            continue
        if _word_bounded(k, key) and len(k) >= 2 and len(k) <= len(key):
            cand = (0, 0, -len(k))
        elif k.startswith(key + " "):
            rest = k[len(key) + 1 :]
            sub = 40
            for suf, pri in _DEF_SUFFIX.items():
                if rest == suf or rest.startswith(suf + " "):
                    sub = pri
                    break
            # definitional suffixes first; among others prefer longer (more specific)
            cand = (1, sub, -len(k) if sub < 40 else len(k))
        elif multi and _word_bounded(key, k) and len(key) >= 5:
            cand = (2, 0, len(k))
        else:
            continue
        if best is None or cand < best:
            best = cand
            best_v = v
    if best is not None:
        return best_v
    return f"ERROR: unknown key ({key!r}); try capital france, pi, water formula"


def tool_wiki(*args: str, **kwargs: str) -> str:
    """Short Wikipedia summary (network). Caps length for T=256."""
    q = kwargs.get("q") or " ".join(args)
    q = q.strip()
    if not q:
        return "ERROR: empty query"
    title = urllib.parse.quote(q.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ll3m-agent/0.1 (local research; educational)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # search API fallback
            return _wiki_search(q)
        return f"ERROR: http {e.code}"
    except Exception as e:
        return f"ERROR: {e}"
    extract = (data.get("extract") or data.get("description") or "").strip()
    if not extract:
        return _wiki_search(q)
    if len(extract) > 400:
        extract = extract[:397] + "..."
    return extract


def _wiki_search(q: str) -> str:
    params = urllib.parse.urlencode(
        {"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"}
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "ll3m-agent/0.1 (local research; educational)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        if len(data) >= 2 and data[1]:
            return tool_wiki(data[1][0])
    except Exception as e:
        return f"ERROR: search failed ({e})"
    return "ERROR: no wiki page found"


def tool_note(*args: str, **kwargs: str) -> str:
    text = kwargs.get("text") or " ".join(args)
    if not text:
        return "ERROR: empty note"
    return f"noted:{text[:200]}"


DEFAULT_TOOLS: dict[str, ToolSpec] = {
    "calc": ToolSpec("calc", "calc op a b | calc expr=1+2*3", tool_calc),
    "time": ToolSpec("time", "time  (UTC now)", tool_time),
    "lookup": ToolSpec("lookup", "lookup key  (closed KB)", tool_lookup),
    "wiki": ToolSpec("wiki", "wiki topic  (Wikipedia summary)", tool_wiki),
    "note": ToolSpec("note", "note free text", tool_note),
}


def tools_brief(tools: dict[str, ToolSpec] | None = None) -> str:
    tools = tools or DEFAULT_TOOLS
    lines = ["Tools (emit one call then wait for result):"]
    for t in tools.values():
        lines.append(f"  {t.name}: {t.doc}")
    lines.append(f"Format: {CALL_OPEN}name args{CALL_CLOSE}")
    lines.append("If no tool needed, answer in plain text only.")
    return "\n".join(lines)


def parse_call(text: str) -> tuple[str, list[str], dict[str, str]] | None:
    """Return (name, positional_args, kwargs) or None if no call tag."""
    if not text:
        return None
    m = _CALL_RE.search(text)
    if not m:
        return None
    name = m.group(1).lower()
    if not name:
        return None
    rest = m.group(2).strip()
    pos: list[str] = []
    kw: dict[str, str] = {}
    if rest:
        # tokenize: key=value or bare tokens (quoted strings kept as one arg)
        for tok in re.findall(r'[^\s"]+|"[^"]*"', rest):
            tok = tok.strip('"')
            if not tok:
                continue
            if "=" in tok:
                k, _, v = tok.partition("=")
                k = k.lower()
                if not k:
                    continue  # bare "=foo" is not a valid kw
                kw[k] = v
            else:
                pos.append(tok)
    return name, pos, kw


def format_result(body: str, max_chars: int = 400) -> str:
    body = body.replace("\n", " ").strip()
    if len(body) > max_chars:
        body = body[: max_chars - 3] + "..."
    return f"{RESULT_OPEN}{body}{RESULT_CLOSE}"


def execute_call(
    name: str,
    args: list[str],
    kwargs: dict[str, str],
    tools: dict[str, ToolSpec] | None = None,
) -> str:
    tools = tools or DEFAULT_TOOLS
    spec = tools.get(name)
    if spec is None:
        return f"ERROR: unknown tool {name!r} (have: {', '.join(sorted(tools))})"
    try:
        return spec.run(*args, **kwargs)
    except TypeError as e:
        return f"ERROR: bad args ({e})"
    except Exception as e:
        return f"ERROR: {e}"


def extract_call_span(text: str) -> str | None:
    m = _CALL_RE.search(text)
    return m.group(0) if m else None
