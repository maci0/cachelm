"""Stage 2: rule-based claim extractors (P1 mechanical)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ll3m.semcomp.schema import (
    EXTRACTOR_RULE_V1,
    MEANING_MODEL_HARD_V1,
    Claim,
    LiteralBundle,
    SourceSpan,
    claim_id_for,
)
from ll3m.semcomp.segment import Document, Segment, segment_sentences

_NUM = re.compile(
    r"(?<![\w.])([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![\w.])"
)
_UNIT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(°C|°F|celsius|fahrenheit|kelvin|K|km|m|cm|mm|kg|g|mg|mol|s|ms|Hz|kHz|MHz|"
    r"J|kJ|W|V|A|Pa|atm|%|percent)\b",
    re.I,
)
_NEG = re.compile(r"\b(not|no|never|cannot|can't|n't|without)\b", re.I)

_IS_OF = re.compile(
    r"^(?:the\s+)?(?P<sub>.+?)\s+of\s+(?P<ent>.+?)\s+is\s+(?P<obj>.+?)\.?$",
    re.I,
)
_AT = re.compile(
    r"^(?P<sub>.+?)\s+(?P<pred>freezes|boils|melts|occurs|happens)\s+at\s+(?P<obj>.+?)\.?$",
    re.I,
)
_IS = re.compile(
    r"^(?P<sub>[A-Z][^.]{0,80}?)\s+is\s+(?P<obj>.+?)\.?$",
)
_QA = re.compile(
    r"^(?:q(?:uestion)?|prompt)\s*[:：]\s*(?P<q>.+?)\s+"
    r"(?:a(?:nswer)?|response)\s*[:：]\s*(?P<a>.+)$",
    re.I | re.S,
)
_ANSWER_LINE = re.compile(r"^(?:answer|a)\s*[:：]\s*(?P<a>.+)$", re.I)
_QUESTION_LINE = re.compile(r"^(?:question|q)\s*[:：]\s*(?P<q>.+)$", re.I)

_USER = re.compile(r"^(?:user|human)\s*[:：]\s*(?P<u>.+)$", re.I)
_ASSIST = re.compile(r"^(?:assistant|bot|model)\s*[:：]\s*(?P<a>.+)$", re.I)
_TOOL_LINE = re.compile(r"^(?:tool|result)\s*[:：]\s*(?P<t>.+)$", re.I)

_CALL = re.compile(r"<call>\s*(?P<body>.*?)\s*</call>", re.I | re.S)
_RESULT = re.compile(r"<result>\s*(?P<body>.*?)\s*</result>", re.I | re.S)

# "Define X." / "What is X?" / "What is capital of utah?"
_DEFINE = re.compile(
    r"^(?:define|what is|what's|tell me about(?: what is)?)\s+(?P<topic>.+?)\??$",
    re.I,
)

# System / boilerplate to skip as docs of claims
_SKIP_LINE = re.compile(
    r"^(you are a small helper|tools\s*\(|format:\s*<call>|if no tool needed|"
    r"once upon a time|click here|subscribe to our|please rate)",
    re.I,
)


def extract_literals(text: str) -> LiteralBundle:
    # preserve order, drop duplicate number/unit strings
    nums: list[str] = []
    for n in _NUM.findall(text):
        if n not in nums:
            nums.append(n)
    units: list[str] = []
    for a, b in _UNIT.findall(text):
        u = f"{a} {b}"
        if u not in units:
            units.append(u)
    ents = tuple(
        sorted(
            {
                m.group(0)
                for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            }
        )
    )
    neg = bool(_NEG.search(text))
    return LiteralBundle(numbers=tuple(nums), units=tuple(units), entities=ents, negation=neg)


def extract_from_segment(seg: Segment) -> list[Claim]:
    t = seg.text.strip()
    if not t or t.startswith("#") or _SKIP_LINE.match(t):
        return []
    claims: list[Claim] = []

    for m in _CALL.finditer(t):
        body = re.sub(r"\s+", " ", m.group("body").strip())
        surface = f"<call>{body}</call>"
        lit = extract_literals(body)
        subj, _, obj = body.partition(" ")
        claims.append(
            _make(
                ctype="tool_trace",
                subject=subj or "tool",
                predicate="tool_call",
                obj=obj or body,
                surface=surface,
                lit=lit,
                seg=seg,
            )
        )

    m = _QA.match(t)
    if m:
        q, a = m.group("q").strip(), m.group("a").strip()
        lit = extract_literals(f"{q} {a}")
        claims.append(
            _make(
                ctype="qa",
                subject=q,
                predicate="answer",
                obj=a,
                surface=f"Q: {q} A: {a}",
                lit=lit,
                seg=seg,
            )
        )
        return claims

    m = _IS_OF.match(t)
    if m:
        sub = f"{m.group('sub').strip()} of {m.group('ent').strip()}"
        obj = m.group("obj").strip().rstrip(".")
        lit = extract_literals(t)
        claims.append(
            _make(
                ctype="fact",
                subject=sub,
                predicate="is",
                obj=obj,
                surface=t if t.endswith(".") else t + ".",
                lit=lit,
                seg=seg,
            )
        )
        return claims

    m = _AT.match(t)
    if m:
        lit = extract_literals(t)
        claims.append(
            _make(
                ctype="fact",
                subject=m.group("sub").strip(),
                predicate=m.group("pred").strip().lower(),
                obj=m.group("obj").strip().rstrip("."),
                surface=t if t.endswith(".") else t + ".",
                lit=lit,
                seg=seg,
            )
        )
        return claims

    m = _IS.match(t)
    if m and " of " not in m.group("sub").lower():
        lit = extract_literals(t)
        claims.append(
            _make(
                ctype="definition",
                subject=m.group("sub").strip(),
                predicate="is",
                obj=m.group("obj").strip().rstrip("."),
                surface=t if t.endswith(".") else t + ".",
                lit=lit,
                seg=seg,
            )
        )
        return claims

    lit = extract_literals(t)
    if lit.numbers or lit.units or lit.negation:
        claims.append(
            _make(
                ctype="fact",
                subject=t[:80],
                predicate="states",
                obj=t,
                surface=t if t.endswith((".", "!", "?")) else t + ".",
                lit=lit,
                seg=seg,
                confidence=0.5,
            )
        )
    return claims


def extract_from_document(doc: Document) -> list[Claim]:
    """Line-aware extract for Q/A and User/Assistant dialogues, plus sentence rules."""
    out: list[Claim] = []
    text = doc.text
    lines = text.splitlines()
    # map line index -> char offset
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#") or _SKIP_LINE.match(line):
            i += 1
            continue

        # Q: / A: on consecutive lines
        mq = _QUESTION_LINE.match(line)
        if mq and i + 1 < len(lines):
            ma = _ANSWER_LINE.match(lines[i + 1].strip())
            if ma:
                q, a = mq.group("q").strip(), ma.group("a").strip()
                start = offsets[i]
                end = offsets[i + 1] + len(lines[i + 1])
                out.append(
                    _make_span(
                        doc,
                        start,
                        end,
                        ctype="qa",
                        subject=q,
                        predicate="answer",
                        obj=a,
                        surface=f"Q: {q} A: {a}",
                    )
                )
                i += 2
                continue

        # User: ... Assistant: ... [Tool: ...] [Assistant: ...]
        mu = _USER.match(line)
        if mu and i + 1 < len(lines):
            user_q = mu.group("u").strip()
            # collect following assist/tool until next User/Q or blank block end
            j = i + 1
            assist_parts: list[str] = []
            tool_parts: list[str] = []
            calls: list[str] = []
            while j < len(lines):
                lj = lines[j].strip()
                if not lj:
                    if assist_parts or tool_parts or calls:
                        break
                    j += 1
                    continue
                if _USER.match(lj) or _QUESTION_LINE.match(lj):
                    break
                if _SKIP_LINE.match(lj):
                    j += 1
                    continue
                ma = _ASSIST.match(lj)
                if ma:
                    body = ma.group("a").strip()
                    assist_parts.append(body)
                    for cm in _CALL.finditer(body):
                        calls.append(cm.group("body").strip())
                    j += 1
                    continue
                mt = _TOOL_LINE.match(lj)
                if mt:
                    tb = mt.group("t").strip()
                    tool_parts.append(tb)
                    for rm in _RESULT.finditer(tb):
                        tool_parts.append(rm.group("body").strip())
                    j += 1
                    continue
                # bare tool/result lines sometimes
                if lj.startswith("<result>") or lj.startswith("Tool:"):
                    tool_parts.append(lj)
                    j += 1
                    continue
                break

            end_line = max(i, j - 1)
            start = offsets[i]
            end = offsets[end_line] + len(lines[end_line])
            final_ans = ""
            for a in reversed(assist_parts):
                if not _CALL.search(a):
                    final_ans = a
                    break
            if not final_ans and tool_parts:
                # use last result body
                last = tool_parts[-1]
                rm = _RESULT.search(last)
                final_ans = rm.group("body").strip() if rm else last

            # tool traces
            for body in calls:
                body_n = re.sub(r"\s+", " ", body)
                out.append(
                    _make_span(
                        doc,
                        start,
                        end,
                        ctype="tool_trace",
                        subject=body_n.split()[0] if body_n.split() else "tool",
                        predicate="tool_call",
                        obj=body_n,
                        surface=f"<call>{body_n}</call>",
                    )
                )

            # Q/A from user + final answer
            topic = user_q
            dm = _DEFINE.match(user_q)
            if dm:
                topic = dm.group("topic").strip().rstrip("?.")
            if final_ans and not _CALL.search(final_ans):
                out.append(
                    _make_span(
                        doc,
                        start,
                        end,
                        ctype="qa",
                        subject=topic,
                        predicate="answer",
                        obj=final_ans.rstrip("."),
                        surface=f"Q: {topic} A: {final_ans.rstrip('.')}",
                    )
                )
            i = j if j > i + 1 else i + 1
            continue

        # single-line Q: ... A: ...
        seg = Segment(doc_id=doc.doc_id, start=offsets[i], end=offsets[i] + len(lines[i]), text=line)
        out.extend(extract_from_segment(seg))
        i += 1

    # Also run sentence pass for free prose not covered by dialogue (skip long story-like)
    if len(text) < 800 and "once upon a time" not in text.lower():
        for seg in segment_sentences(doc):
            # skip if already fully covered as substring of existing surfaces
            if any(seg.text in c.surface or seg.text in (c.source_spans[0].text if c.source_spans else "") for c in out):
                continue
            if _USER.match(seg.text) or _ASSIST.match(seg.text) or _QUESTION_LINE.match(seg.text):
                continue
            out.extend(extract_from_segment(seg))

    return out


def extract_corpus(docs: Iterable[Document]) -> list[Claim]:
    claims: list[Claim] = []
    for d in docs:
        claims.extend(extract_from_document(d))
    return claims


def _make(
    *,
    ctype: str,
    subject: str,
    predicate: str,
    obj: str,
    surface: str,
    lit: LiteralBundle,
    seg: Segment,
    confidence: float = 1.0,
) -> Claim:
    span = SourceSpan(doc_id=seg.doc_id, start=seg.start, end=seg.end, text=seg.text)
    c = Claim(
        claim_id="",
        type=ctype,  # type: ignore[arg-type]
        subject=subject,
        predicate=predicate,
        object=obj,
        surface=surface,
        literals=lit,
        source_spans=[span],
        confidence=confidence,
        extractor=EXTRACTOR_RULE_V1,
        meaning_model=MEANING_MODEL_HARD_V1,
    )
    c.claim_id = claim_id_for(c.canonical_key(), c.surface, seg.doc_id)
    return c


def _make_span(
    doc: Document,
    start: int,
    end: int,
    *,
    ctype: str,
    subject: str,
    predicate: str,
    obj: str,
    surface: str,
    confidence: float = 1.0,
) -> Claim:
    span_text = doc.text[start:end]
    lit = extract_literals(f"{subject} {obj} {surface}")
    span = SourceSpan(doc_id=doc.doc_id, start=start, end=end, text=span_text)
    c = Claim(
        claim_id="",
        type=ctype,  # type: ignore[arg-type]
        subject=subject,
        predicate=predicate,
        object=obj,
        surface=surface,
        literals=lit,
        source_spans=[span],
        confidence=confidence,
        extractor=EXTRACTOR_RULE_V1,
        meaning_model=MEANING_MODEL_HARD_V1,
    )
    c.claim_id = claim_id_for(c.canonical_key(), c.surface, doc.doc_id)
    return c
