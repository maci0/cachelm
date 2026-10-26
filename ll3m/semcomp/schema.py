"""Hard meaning model M* v1 and serializable claim schema."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ClaimType = Literal["fact", "procedure", "tool_trace", "definition", "constraint", "qa"]

MEANING_MODEL_HARD_V1 = "hard_v1"
EXTRACTOR_RULE_V1 = "rule@v1"

# Content words kept for soft lexical checks (not the hard key).
_STOP = frozenset(
    "a an the is are was were be been being of to in on for and or but with as by at from that this it".split()
)


@dataclass(frozen=True)
class SourceSpan:
    doc_id: str
    start: int
    end: int
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SourceSpan:
        return SourceSpan(
            doc_id=str(d["doc_id"]),
            start=int(d["start"]),
            end=int(d["end"]),
            text=str(d.get("text") or ""),
        )


@dataclass(frozen=True)
class LiteralBundle:
    numbers: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    negation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "numbers": list(self.numbers),
            "units": list(self.units),
            "entities": list(self.entities),
            "negation": self.negation,
        }

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> LiteralBundle:
        if not d:
            return LiteralBundle()
        return LiteralBundle(
            numbers=tuple(str(x) for x in d.get("numbers") or ()),
            units=tuple(str(x) for x in d.get("units") or ()),
            entities=tuple(str(x) for x in d.get("entities") or ()),
            negation=bool(d.get("negation") or False),
        )


@dataclass
class Claim:
    """One hard meaning unit under M* hard_v1."""

    claim_id: str
    type: ClaimType
    subject: str
    predicate: str
    object: str
    surface: str
    literals: LiteralBundle = field(default_factory=LiteralBundle)
    source_spans: list[SourceSpan] = field(default_factory=list)
    confidence: float = 1.0
    extractor: str = EXTRACTOR_RULE_V1
    meaning_model: str = MEANING_MODEL_HARD_V1
    conflict: bool = False

    def canonical_key(self) -> str:
        """Stable key for coverage / dedup (not the free-form surface)."""
        neg = "1" if self.literals.negation else "0"
        nums = ",".join(self.literals.numbers)
        units = ",".join(self.literals.units)
        parts = [
            self.type,
            _norm_key(self.subject),
            _norm_key(self.predicate),
            _norm_key(self.object),
            neg,
            nums,
            units,
        ]
        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "type": self.type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "surface": self.surface,
            "literals": self.literals.to_dict(),
            "source_spans": [s.to_dict() for s in self.source_spans],
            "confidence": self.confidence,
            "extractor": self.extractor,
            "meaning_model": self.meaning_model,
            "conflict": self.conflict,
            "canonical_key": self.canonical_key(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Claim:
        return Claim(
            claim_id=str(d["claim_id"]),
            type=d["type"],  # type: ignore[arg-type]
            subject=str(d.get("subject") or ""),
            predicate=str(d.get("predicate") or ""),
            object=str(d.get("object") or ""),
            surface=str(d.get("surface") or ""),
            literals=LiteralBundle.from_dict(d.get("literals")),
            source_spans=[SourceSpan.from_dict(s) for s in d.get("source_spans") or []],
            confidence=float(d.get("confidence") or 1.0),
            extractor=str(d.get("extractor") or EXTRACTOR_RULE_V1),
            meaning_model=str(d.get("meaning_model") or MEANING_MODEL_HARD_V1),
            conflict=bool(d.get("conflict") or False),
        )


@dataclass
class CoverageReport:
    meaning_model: str
    n_source_claims: int
    n_output_claims: int
    n_covered: int
    n_missing: int
    n_invented: int
    missing_keys: list[str]
    invented_keys: list[str]
    literal_failures: list[str]
    token_in: int | None = None
    token_out: int | None = None
    char_in: int = 0
    char_out: int = 0
    ok: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.token_in and self.token_out is not None and self.token_in > 0:
            d["token_ratio"] = self.token_out / self.token_in
        if self.char_in > 0:
            d["char_ratio"] = self.char_out / self.char_in
        return d


def claim_id_for(key: str, surface: str, doc_id: str) -> str:
    h = hashlib.sha1(f"{key}\n{surface}\n{doc_id}".encode("utf-8")).hexdigest()
    return h[:16]


def _norm_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s.\-/%°]", "", s)
    return s


def content_tokens(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())
    return {t for t in toks if t not in _STOP and len(t) > 1}


def dumps_claims(claims: list[Claim], *, indent: int | None = 2) -> str:
    return json.dumps([c.to_dict() for c in claims], indent=indent, ensure_ascii=False) + "\n"


def loads_claims(raw: str | bytes) -> list[Claim]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("claims JSON must be a list")
    return [Claim.from_dict(x) for x in data]
