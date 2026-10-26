"""Semantic compressor: densify training text under a hard meaning model.

See docs/SEMANTIC_COMPRESSOR.md. P0/P1: rule extract + claim dedup + verify.
"""

from ll3m.semcomp.schema import (
    MEANING_MODEL_HARD_V1,
    Claim,
    CoverageReport,
    LiteralBundle,
    SourceSpan,
)

__all__ = [
    "MEANING_MODEL_HARD_V1",
    "Claim",
    "CoverageReport",
    "LiteralBundle",
    "SourceSpan",
]
