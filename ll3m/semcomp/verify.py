"""Stage 5: coverage + no-invention + literal checks."""

from __future__ import annotations

import re

from ll3m.semcomp.schema import MEANING_MODEL_HARD_V1, Claim, CoverageReport


def verify_claims(
    source: list[Claim],
    output: list[Claim],
    *,
    meaning_model: str = MEANING_MODEL_HARD_V1,
    char_in: int = 0,
    char_out: int = 0,
    token_in: int | None = None,
    token_out: int | None = None,
) -> CoverageReport:
    src_keys = {c.canonical_key(): c for c in source}
    out_keys = {c.canonical_key(): c for c in output}

    missing = sorted(k for k in src_keys if k not in out_keys)
    invented = sorted(k for k in out_keys if k not in src_keys)

    lit_fail: list[str] = []
    for k, sc in src_keys.items():
        oc = out_keys.get(k)
        if oc is None:
            continue
        lit_fail.extend(_literal_check(sc, oc))

    n_src = len(src_keys)
    n_out = len(out_keys)
    n_cov = n_src - len(missing)
    ok = len(missing) == 0 and len(invented) == 0 and len(lit_fail) == 0

    notes: list[str] = []
    if any(c.conflict for c in output):
        notes.append("conflicts present (both sides kept)")
    if any(c.confidence < 1.0 for c in source):
        notes.append("source has low-confidence fallback claims")

    return CoverageReport(
        meaning_model=meaning_model,
        n_source_claims=n_src,
        n_output_claims=n_out,
        n_covered=n_cov,
        n_missing=len(missing),
        n_invented=len(invented),
        missing_keys=missing,
        invented_keys=invented,
        literal_failures=lit_fail,
        token_in=token_in,
        token_out=token_out,
        char_in=char_in,
        char_out=char_out,
        ok=ok,
        notes=notes,
    )


def reextract_and_verify(
    source_claims: list[Claim],
    rendered_text: str,
    *,
    reextract_fn,
    doc_id: str = "rendered",
    char_in: int = 0,
    token_in: int | None = None,
    token_out: int | None = None,
) -> CoverageReport:
    """Round-trip: re-extract M* from rendered surface and compare to source claims."""
    from ll3m.semcomp.segment import Document

    docs = [Document(doc_id=doc_id, text=rendered_text if rendered_text.endswith("\n") else rendered_text + "\n")]
    out_claims = reextract_fn(docs)
    return verify_claims(
        source_claims,
        out_claims,
        char_in=char_in,
        char_out=len(rendered_text),
        token_in=token_in,
        token_out=token_out,
    )


def _literal_check(src: Claim, out: Claim) -> list[str]:
    fails: list[str] = []
    surface = out.surface
    for n in src.literals.numbers:
        if n and n not in surface and not _num_present(n, surface):
            fails.append(f"{src.claim_id}: missing number {n!r}")
    for u in src.literals.units:
        # unit strings like "0 °C" — require the unit token
        unit_tok = u.split()[-1] if u.split() else u
        if unit_tok and unit_tok not in surface and unit_tok.lower() not in surface.lower():
            # degrees Celsius spelled out
            if unit_tok in ("°C", "celsius") and re.search(r"celsius|°C", surface, re.I):
                continue
            fails.append(f"{src.claim_id}: missing unit {u!r}")
    if src.literals.negation and not re.search(
        r"\b(not|no|never|cannot|can't|n't|without)\b", surface, re.I
    ):
        fails.append(f"{src.claim_id}: missing negation")
    return fails


def _num_present(n: str, surface: str) -> bool:
    # allow 0 vs 0.0 mild variants
    if n in surface:
        return True
    try:
        f = float(n)
    except ValueError:
        return False
    for m in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", surface):
        try:
            if abs(float(m) - f) < 1e-9:
                return True
        except ValueError:
            continue
    return False
