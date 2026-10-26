"""Orchestrate normalize → extract → dedup → render → verify."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ll3m.semcomp.dedup import dedup_claims
from ll3m.semcomp.extract import extract_corpus
from ll3m.semcomp.normalize import normalize_text
from ll3m.semcomp.render import RenderProfile, render_claims
from ll3m.semcomp.schema import MEANING_MODEL_HARD_V1, Claim, CoverageReport, dumps_claims
from ll3m.semcomp.segment import Document, load_documents
from ll3m.semcomp.verify import reextract_and_verify, verify_claims


def _count_tokens(text: str, tokenizer_path: str | None) -> int | None:
    if not tokenizer_path:
        # whitespace proxy
        return len(text.split())
    try:
        from ll3m.tokenizer import BPETokenizer

        tok = BPETokenizer.load(tokenizer_path)
        return len(tok.encode(text))
    except Exception:
        return len(text.split())


def load_normalized_docs(path: str | Path) -> list[Document]:
    docs = load_documents(path)
    return [
        Document(doc_id=d.doc_id, text=normalize_text(d.text), path=d.path) for d in docs
    ]


def run_pipeline(
    in_path: str | Path,
    *,
    profile: RenderProfile = "atomic",
    meaning_model: str = MEANING_MODEL_HARD_V1,
    tokenizer_path: str | None = None,
    roundtrip: bool = True,
) -> dict[str, Any]:
    docs = load_normalized_docs(in_path)
    raw_text = "\n".join(d.text for d in docs)
    source_claims = extract_corpus(docs)
    claims = dedup_claims(list(source_claims))
    # train mix = surfaces only (no # claim headers); audit copy keeps headers
    rendered = render_claims(claims, profile=profile, with_headers=False)
    rendered_audit = render_claims(claims, profile=profile, with_headers=True)
    if not rendered.endswith("\n"):
        rendered += "\n"
    if not rendered_audit.endswith("\n"):
        rendered_audit += "\n"

    tok_in = _count_tokens(raw_text, tokenizer_path)
    tok_out = _count_tokens(rendered, tokenizer_path)

    # identity verify on claim lists (dedup must not drop unique keys)
    report = verify_claims(
        source_claims,
        claims,
        meaning_model=meaning_model,
        char_in=len(raw_text),
        char_out=len(rendered),
        token_in=tok_in,
        token_out=tok_out,
    )

    rt: CoverageReport | None = None
    if roundtrip:
        # Round-trip: re-extract from train surface; compare by canonical key
        rt = reextract_and_verify(
            claims,
            rendered,
            reextract_fn=lambda ds: dedup_claims(extract_corpus(ds)),
            char_in=len(raw_text),
            token_in=tok_in,
            token_out=tok_out,
        )
        report.notes.append(f"roundtrip_ok={rt.ok}")
        # Strict claim-list coverage is the hard gate; roundtrip is advisory for P1
        # (rule extract is not fully invertible on all surface styles).
        if not rt.ok:
            report.notes.append(
                f"roundtrip residual missing={rt.n_missing} invented={rt.n_invented} "
                f"lit={len(rt.literal_failures)} (advisory; claim-list gate separate)"
            )
            report.notes.append(
                "roundtrip_keys_missing_sample="
                + ",".join(rt.missing_keys[:3])
            )

    return {
        "docs": [{"doc_id": d.doc_id, "path": d.path, "chars": len(d.text)} for d in docs],
        "claims": claims,
        "source_claim_count": len(source_claims),
        "dedup_claim_count": len(claims),
        "rendered": rendered,
        "rendered_audit": rendered_audit,
        "report": report,
        "roundtrip": rt,
        "profile": profile,
        "meaning_model": meaning_model,
    }


def write_run_output(
    result: dict[str, Any],
    out_dir: str | Path,
    *,
    refuse_on_residual: bool = True,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: CoverageReport = result["report"]
    claims: list[Claim] = result["claims"]

    (out / "claims.json").write_text(dumps_claims(claims), encoding="utf-8")
    (out / "coverage.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    if result.get("roundtrip") is not None:
        (out / "roundtrip.json").write_text(
            json.dumps(result["roundtrip"].to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    rt = result.get("roundtrip")
    manifest = {
        "meaning_model": result["meaning_model"],
        "profile": result["profile"],
        "n_docs": len(result["docs"]),
        "n_claims": len(claims),
        "coverage_ok": report.ok,
        "roundtrip_ok": bool(rt.ok) if rt is not None else None,
        "docs": result["docs"],
        "report_summary": {
            "n_source_claims": report.n_source_claims,
            "n_output_claims": report.n_output_claims,
            "n_missing": report.n_missing,
            "n_invented": report.n_invented,
            "token_in": report.token_in,
            "token_out": report.token_out,
            "char_ratio": (report.char_out / report.char_in) if report.char_in else None,
            "token_ratio": (report.token_out / report.token_in)
            if report.token_in and report.token_out
            else None,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if refuse_on_residual and not report.ok:
        (out / "REFUSED.txt").write_text(
            "refuse-on-residual: claim-list coverage not ok; mix not written\n"
            + json.dumps(report.to_dict(), indent=2)
            + "\n",
            encoding="utf-8",
        )
        return out

    (out / "mix.txt").write_text(result["rendered"], encoding="utf-8")
    if "rendered_audit" in result:
        (out / "mix_audit.txt").write_text(result["rendered_audit"], encoding="utf-8")
    return out


def estimate_only(in_path: str | Path, *, tokenizer_path: str | None = None) -> dict[str, Any]:
    r = run_pipeline(in_path, profile="atomic", tokenizer_path=tokenizer_path, roundtrip=False)
    rep: CoverageReport = r["report"]
    return {
        "n_docs": len(r["docs"]),
        "n_claims_raw": r["source_claim_count"],
        "n_claims_dedup": r["dedup_claim_count"],
        "char_in": rep.char_in,
        "char_out": rep.char_out,
        "char_ratio": (rep.char_out / rep.char_in) if rep.char_in else None,
        "token_in": rep.token_in,
        "token_out": rep.token_out,
        "token_ratio": (rep.token_out / rep.token_in)
        if rep.token_in and rep.token_out is not None and rep.token_in
        else None,
        "conflicts": sum(1 for c in r["claims"] if c.conflict),
    }


def audit_output(corpus_dir: str | Path, against: str | Path) -> CoverageReport:
    """Re-extract from compressed mix and compare to claims extracted from source."""
    corp = Path(corpus_dir)
    mix = corp / "mix.txt"
    if not mix.exists():
        raise FileNotFoundError(f"missing {mix}")
    src_docs = load_normalized_docs(against)
    src_claims = dedup_claims(extract_corpus(src_docs))
    rendered = mix.read_text(encoding="utf-8")
    return reextract_and_verify(
        src_claims,
        rendered,
        reextract_fn=lambda ds: dedup_claims(extract_corpus(ds)),
        char_in=sum(len(d.text) for d in src_docs),
    )
