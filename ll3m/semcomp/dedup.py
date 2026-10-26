"""Stage 3: claim-level dedup / conflict mark."""

from __future__ import annotations

from ll3m.semcomp.schema import Claim


def dedup_claims(claims: list[Claim]) -> list[Claim]:
    """Keep first claim per canonical_key; mark conflicts on SPO clash with different object.

    Conflict rule: same (type, subject, predicate, negation) but different object/numbers
    keeps both with conflict=True (never silent drop).
    """
    by_key: dict[str, Claim] = {}
    spo_index: dict[tuple[str, str, str, bool], list[Claim]] = {}
    ordered: list[Claim] = []

    for c in claims:
        key = c.canonical_key()
        if key in by_key:
            # exact same claim: merge source spans
            prev = by_key[key]
            seen = {(s.doc_id, s.start, s.end) for s in prev.source_spans}
            for s in c.source_spans:
                t = (s.doc_id, s.start, s.end)
                if t not in seen:
                    prev.source_spans.append(s)
                    seen.add(t)
            continue

        spo = (
            c.type,
            c.subject.strip().lower(),
            c.predicate.strip().lower(),
            c.literals.negation,
        )
        rivals = spo_index.get(spo, [])
        for r in rivals:
            if r.object.strip().lower() != c.object.strip().lower():
                r.conflict = True
                c.conflict = True

        by_key[key] = c
        spo_index.setdefault(spo, []).append(c)
        ordered.append(c)

    return ordered
