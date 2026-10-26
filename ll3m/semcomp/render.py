"""Stage 4: deterministic surfaces from claim sets."""

from __future__ import annotations

from typing import Literal

from ll3m.semcomp.schema import Claim

RenderProfile = Literal["atomic", "qa", "compact_prose", "mixed"]


def render_claims(
    claims: list[Claim],
    profile: RenderProfile = "atomic",
    *,
    with_headers: bool = False,
) -> str:
    """Render claim surfaces for train (default) or audit (with_headers=True)."""
    if profile == "atomic":
        return _render_atomic(claims, with_headers=with_headers)
    if profile == "qa":
        return _render_qa(claims)
    if profile == "compact_prose":
        return _render_prose(claims)
    if profile == "mixed":
        facts = [c for c in claims if c.type in ("fact", "definition", "constraint", "procedure")]
        qas = [c for c in claims if c.type == "qa"]
        tools = [c for c in claims if c.type == "tool_trace"]
        parts = [
            _render_atomic(facts, with_headers=with_headers),
            _render_qa(qas),
            _render_atomic(tools, with_headers=with_headers),
        ]
        return "\n".join(p for p in parts if p.strip()) + ("\n" if any(parts) else "")
    raise ValueError(f"unknown profile: {profile}")


def _render_atomic(claims: list[Claim], *, with_headers: bool = False) -> str:
    lines: list[str] = []
    for c in claims:
        if with_headers:
            src = ",".join(sorted({s.doc_id for s in c.source_spans})) or "?"
            flag = " conflict" if c.conflict else ""
            lines.append(f"# claim:{c.claim_id} src:{src}{flag}")
        surf = c.surface.rstrip()
        lines.append(surf)
    body = "\n".join(lines)
    return body + ("\n" if body else "")


def _render_qa(claims: list[Claim]) -> str:
    lines: list[str] = []
    for c in claims:
        if c.type != "qa":
            # promote fact to synthetic Q
            q = f"What is true about {c.subject}?"
            a = c.object if c.predicate == "is" else c.surface
            lines.append(f"Question: {q}")
            lines.append(f"Answer: {a}")
            lines.append("")
            continue
        lines.append(f"Question: {c.subject}")
        lines.append(f"Answer: {c.object}")
        lines.append("")
    return "\n".join(lines)


def _render_prose(claims: list[Claim], *, pack: int = 4) -> str:
    """Pack up to `pack` claim surfaces into short paragraphs."""
    surfaces = [c.surface.rstrip().rstrip(".") + "." for c in claims]
    paras: list[str] = []
    for i in range(0, len(surfaces), pack):
        chunk = surfaces[i : i + pack]
        paras.append(" ".join(chunk))
    return "\n\n".join(paras) + ("\n" if paras else "")
