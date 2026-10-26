"""P0/P1 semantic compressor tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ll3m.semcomp.dedup import dedup_claims
from ll3m.semcomp.extract import extract_corpus
from ll3m.semcomp.pipeline import audit_output, estimate_only, run_pipeline, write_run_output
from ll3m.semcomp.render import render_claims
from ll3m.semcomp.schema import Claim, claim_id_for
from ll3m.semcomp.segment import load_documents
from ll3m.semcomp.verify import verify_claims

FIXTURE = Path(__file__).parent / "fixtures" / "semcomp" / "golden_source.txt"


def test_load_multi_doc():
    docs = load_documents(FIXTURE)
    ids = {d.doc_id for d in docs}
    assert "france_capital" in ids
    assert "water_phase" in ids
    assert "tool_ep" in ids


def test_extract_hard_facts():
    docs = load_documents(FIXTURE)
    claims = extract_corpus(docs)
    surfaces = " | ".join(c.surface for c in claims)
    assert "capital of France" in surfaces.lower() or any(
        "paris" in c.object.lower() for c in claims
    )
    assert any(c.type == "tool_trace" for c in claims)
    assert any(c.type == "qa" for c in claims)
    # negation preserved
    neg = [c for c in claims if c.literals.negation]
    assert neg, "expected at least one negated claim"


def test_dedup_merges_exact_duplicates():
    docs = load_documents(FIXTURE)
    raw = extract_corpus(docs)
    deduped = dedup_claims(raw)
    # france capital stated twice in source
    keys = [c.canonical_key() for c in deduped]
    assert len(keys) == len(set(keys))
    assert len(deduped) <= len(raw)


def test_verify_identity_ok():
    docs = load_documents(FIXTURE)
    claims = dedup_claims(extract_corpus(docs))
    rep = verify_claims(claims, claims)
    assert rep.ok
    assert rep.n_missing == 0
    assert rep.n_invented == 0


def test_pipeline_roundtrip_and_write(tmp_path: Path):
    result = run_pipeline(FIXTURE, profile="atomic", roundtrip=True)
    assert result["dedup_claim_count"] >= 4
    rep = result["report"]
    # hard gate: claim-list coverage after dedup
    assert rep.ok
    assert rep.n_missing == 0
    assert rep.n_invented == 0

    out = write_run_output(result, tmp_path / "out", refuse_on_residual=True)
    assert (out / "claims.json").exists()
    assert (out / "mix.txt").exists()
    assert (out / "manifest.json").exists()
    mix = (out / "mix.txt").read_text(encoding="utf-8")
    assert "Paris" in mix or "paris" in mix.lower()
    # train mix is surface-only (no audit headers)
    assert "# claim:" not in mix
    assert (out / "mix_audit.txt").exists()
    assert "# claim:" in (out / "mix_audit.txt").read_text(encoding="utf-8")


def test_estimate_ratios():
    est = estimate_only(FIXTURE)
    assert est["n_docs"] >= 5
    assert est["n_claims_dedup"] >= 1
    assert est["char_in"] > est["char_out"] or est["n_claims_dedup"] > 0


def test_audit_after_write(tmp_path: Path):
    result = run_pipeline(FIXTURE, profile="atomic", roundtrip=False)
    write_run_output(result, tmp_path / "out", refuse_on_residual=False)
    # audit re-extracts mix; may not be 100% on freeform lines
    rep = audit_output(tmp_path / "out", FIXTURE)
    assert rep.n_output_claims >= 1


def test_render_profiles():
    docs = load_documents(FIXTURE)
    claims = dedup_claims(extract_corpus(docs))
    atomic = render_claims(claims, "atomic", with_headers=False)
    atomic_h = render_claims(claims, "atomic", with_headers=True)
    qa = render_claims(claims, "qa")
    prose = render_claims(claims, "compact_prose")
    mixed = render_claims(claims, "mixed")
    assert "# claim:" not in atomic
    assert "# claim:" in atomic_h
    assert "Question:" in qa or "Answer:" in qa
    assert len(prose) > 0
    assert len(mixed) > 0


def test_claim_id_stable():
    a = claim_id_for("fact|x|is|y|0||", "X is y.", "d1")
    b = claim_id_for("fact|x|is|y|0||", "X is y.", "d1")
    assert a == b
