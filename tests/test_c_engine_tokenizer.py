"""C engine must use pack-embedded BPE, not legacy story data/tokenizer.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ll3m.c_engine import (
    extract_pack_tokenizer_bytes,
    read_weights_meta,
    resolve_tokenizer_path,
)
from ll3m.tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "checkpoints" / "v76_base_en" / "best.ll3m"
BASE_TOK = ROOT / "data" / "base_en" / "tokenizer.json"
STORY_TOK = ROOT / "data" / "tokenizer.json"


@pytest.mark.skipif(not PACK.is_file(), reason="v76 pack not present")
def test_pack_has_embedded_tokenizer():
    blob = extract_pack_tokenizer_bytes(PACK)
    assert blob is not None and len(blob) > 100
    obj = json.loads(blob.decode("utf-8"))
    assert "stoi" in obj and "merges" in obj
    assert int(obj.get("vocab_size", 0)) == 2048


@pytest.mark.skipif(
    not PACK.is_file() or not BASE_TOK.is_file(),
    reason="v76 pack/tok missing",
)
def test_resolve_prefers_pack_over_story_default(tmp_path: Path):
    # Copy pack into tmp so sidecar write is isolated
    import shutil

    p = tmp_path / "best.ll3m"
    shutil.copy(PACK, p)
    resolved = resolve_tokenizer_path(p)
    assert resolved.is_file()
    pack_tok = BPETokenizer.load(resolved)
    base = BPETokenizer.load(BASE_TOK)
    # Same merges as base_en (not story)
    assert pack_tok.merges[:5] == base.merges[:5]
    if STORY_TOK.is_file():
        story = BPETokenizer.load(STORY_TOK)
        assert pack_tok.merges[:5] != story.merges[:5]


@pytest.mark.skipif(not PACK.is_file(), reason="v76 pack not present")
def test_encode_matches_base_en_not_story():
    from ll3m.c_engine import CEngine

    # Only need encode path; may skip if no ll3m_c binary
    try:
        eng = CEngine(PACK)
    except FileNotFoundError as e:
        if "ll3m_c" in str(e):
            pytest.skip("ll3m_c not built")
        raise
    text = "User: Explain gravity in simple terms.\nAssistant:"
    pack_ids = eng.encode(text)
    if BASE_TOK.is_file():
        base_ids = BPETokenizer.load(BASE_TOK).encode(text)
        assert pack_ids == base_ids
    if STORY_TOK.is_file():
        story_ids = BPETokenizer.load(STORY_TOK).encode(text)
        assert pack_ids != story_ids


def test_read_meta_v76():
    if not PACK.is_file():
        pytest.skip("no pack")
    m = read_weights_meta(PACK)
    assert m.vocab_size == 2048
    assert m.n_layer == 23
    assert m.is_ternary
