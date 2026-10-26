"""Stage 1: segment corpus into docs and sentences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    path: str | None = None


@dataclass(frozen=True)
class Segment:
    doc_id: str
    start: int
    end: int
    text: str


def load_documents(path: str | Path) -> list[Document]:
    """Load .txt files under a dir, or a single file as one/multi-doc.

    Multi-doc file format: lines starting with ``# doc:`` start a new doc.
    Otherwise the whole file is one document.
    """
    p = Path(path)
    if p.is_dir():
        docs: list[Document] = []
        for f in sorted(p.rglob("*.txt")):
            raw = f.read_text(encoding="utf-8")
            docs.extend(_split_file(raw, doc_prefix=f.stem, path=str(f)))
        if not docs:
            raise FileNotFoundError(f"no .txt under {p}")
        return docs
    if not p.is_file():
        raise FileNotFoundError(path)
    return _split_file(p.read_text(encoding="utf-8"), doc_prefix=p.stem, path=str(p))


def _split_file(raw: str, *, doc_prefix: str, path: str) -> list[Document]:
    if re.search(r"(?m)^#\s*doc:", raw):
        docs: list[Document] = []
        cur_id: str | None = None
        buf: list[str] = []
        for line in raw.splitlines():
            m = re.match(r"^#\s*doc:\s*(\S+)\s*$", line.strip())
            if m:
                if cur_id is not None:
                    docs.append(Document(doc_id=cur_id, text="\n".join(buf).strip() + "\n", path=path))
                cur_id = m.group(1)
                buf = []
            else:
                buf.append(line)
        if cur_id is not None:
            docs.append(Document(doc_id=cur_id, text="\n".join(buf).strip() + "\n", path=path))
        return docs
    return [Document(doc_id=doc_prefix, text=raw if raw.endswith("\n") else raw + "\n", path=path)]


def segment_sentences(doc: Document) -> list[Segment]:
    text = doc.text
    segs: list[Segment] = []
    pos = 0
    for m in re.finditer(r"[^\n.!?]+[.!?]?|\n+", text):
        chunk = m.group(0)
        s = chunk.strip()
        if not s:
            continue
        idx = text.find(s, pos)
        if idx < 0:
            idx = m.start()
        end = idx + len(s)
        segs.append(Segment(doc_id=doc.doc_id, start=idx, end=end, text=s))
        pos = end
    if not segs and text.strip():
        s = text.strip()
        segs.append(Segment(doc_id=doc.doc_id, start=0, end=len(s), text=s))
    return segs
