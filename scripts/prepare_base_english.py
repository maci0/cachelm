#!/usr/bin/env python3
"""Build base-English mix + BPE (docs/BASE_ENGLISH.md).

Reads raw texts from data/base_en/raw/, writes:
  data/base_en/mix_train.txt|.bin
  data/base_en/mix_val.txt|.bin
  data/base_en/tokenizer.json
  data/base_en/manifest.json

No Shakespeare. Stories only if present as spice file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from ll3m.tokenizer import train_bpe

_SPLIT_RE = re.compile(r"(?:\n{2,})")
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_END_RE = re.compile(r'[.!?]"?\s*$')
_PRINT_RE = re.compile(r"[^\x09\x0a\x0d\x20-\x7e\u00a0-\u024f]")


def _split_docs(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    return parts


def _quality_ok(
    s: str,
    *,
    min_chars: int,
    max_chars: int,
    min_printable: float,
    max_word_repeat: float,
) -> tuple[bool, str]:
    if len(s) < min_chars:
        return False, "short"
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars]
    printable = 1.0 - (len(_PRINT_RE.findall(s)) / max(len(s), 1))
    if printable < min_printable:
        return False, "printable"
    words = _WORD_RE.findall(s.lower())
    if len(words) >= 20:
        c = Counter(words)
        top = c.most_common(1)[0][1] / len(words)
        if top > max_word_repeat:
            return False, "repeat"
    # allow dialogue without strict sentence end
    if "\n" not in s and not _SENT_END_RE.search(s.strip()):
        if not s.strip().endswith("?"):
            return False, "no_end"
    return True, "ok"


def _norm_hash(s: str) -> str:
    t = re.sub(r"\s+", " ", s.strip().lower())
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def _word_shingles(s: str, n: int = 3) -> set[str]:
    words = _WORD_RE.findall(s.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _minhash_sig(shingles: set[str], num_perm: int = 32) -> tuple[int, ...]:
    """Cheap MinHash: min 32-bit hash per permutation seed."""
    if not shingles:
        return tuple(0 for _ in range(num_perm))
    sig: list[int] = []
    for seed in range(num_perm):
        best = 0xFFFFFFFF
        for sh in shingles:
            h = hashlib.blake2b(f"{seed}:{sh}".encode(), digest_size=4).digest()
            v = int.from_bytes(h, "little")
            if v < best:
                best = v
        sig.append(best)
    return tuple(sig)


def _near_dedup(
    docs: list[str],
    *,
    jaccard_thresh: float = 0.85,
    num_perm: int = 32,
    bands: int = 8,
) -> tuple[list[str], int]:
    """MinHash LSH near-dedup. Drops docs with Jaccard(shingles) >= thresh vs kept.

    Exact-hash dups should already be removed. O(n * band_bucket) typical.
    """
    if not docs or jaccard_thresh <= 0:
        return docs, 0
    rows = num_perm // bands
    if rows < 1:
        rows = 1
        bands = num_perm
    buckets: list[dict[tuple[int, ...], list[int]]] = [dict() for _ in range(bands)]
    kept: list[str] = []
    kept_sh: list[set[str]] = []
    kept_sig: list[tuple[int, ...]] = []
    dropped = 0

    for doc in docs:
        sh = _word_shingles(doc, 3)
        sig = _minhash_sig(sh, num_perm=num_perm)
        cand: set[int] = set()
        for b in range(bands):
            start = b * rows
            key = sig[start : start + rows]
            bucket = buckets[b]
            for idx in bucket.get(key, ()):
                cand.add(idx)
        is_dup = False
        for idx in cand:
            osh = kept_sh[idx]
            inter = len(sh & osh)
            if inter == 0:
                continue
            union = len(sh | osh)
            if union and inter / union >= jaccard_thresh:
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        idx_new = len(kept)
        kept.append(doc)
        kept_sh.append(sh)
        kept_sig.append(sig)
        for b in range(bands):
            start = b * rows
            key = sig[start : start + rows]
            buckets[b].setdefault(key, []).append(idx_new)
    return kept, dropped


def _stratified_sample(docs: list[str], max_chars: int, rng: random.Random) -> str:
    if max_chars <= 0 or not docs:
        return "\n\n".join(docs)
    short = [s for s in docs if len(s) < 500]
    mid = [s for s in docs if 500 <= len(s) < 1500]
    long = [s for s in docs if len(s) >= 1500]
    for b in (short, mid, long):
        rng.shuffle(b)
    budgets = {
        "short": int(max_chars * 0.30),
        "mid": int(max_chars * 0.45),
        "long": int(max_chars * 0.25),
    }
    pieces: list[str] = []
    used = 0
    for name, bucket in (("short", short), ("mid", mid), ("long", long)):
        take = 0
        for s in bucket:
            if take >= budgets[name]:
                break
            pieces.append(s)
            take += len(s) + 2
            used += len(s) + 2
    if used < max_chars:
        rest = short + mid + long
        rng.shuffle(rest)
        for s in rest:
            if used >= max_chars:
                break
            if s in pieces:
                continue
            pieces.append(s)
            used += len(s) + 2
    rng.shuffle(pieces)
    text = "\n\n".join(pieces)
    return text[:max_chars] if len(text) > max_chars else text


def _encode(tok, text: str, out_bin: Path) -> int:
    ids = tok.encode(text)
    arr = np.asarray(ids, dtype=np.uint16 if tok.vocab_size < 65535 else np.int32)
    arr.tofile(out_bin)
    return len(arr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=str, default="data/base_en/raw")
    ap.add_argument("--out-dir", type=str, default="data/base_en")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=60)
    # Prefer short chunks (atomic/edu density); long pages waste capacity.
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--bpe-train-chars", type=int, default=40_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-printable", type=float, default=0.97)
    ap.add_argument("--max-word-repeat", type=float, default=0.20)
    ap.add_argument(
        "--near-dedup-jaccard",
        type=float,
        default=0.85,
        help="MinHash near-dedup threshold (0 disables)",
    )
    ap.add_argument(
        "--prefer-short-weight",
        type=float,
        default=1.6,
        help=">1 upsamples short docs when building train order (density bias)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = sorted(raw.glob("*.txt"))
    if not sources:
        raise SystemExit(f"no raw *.txt in {raw}; run download_base_english.py first")

    docs: list[str] = []
    src_stats: dict[str, int] = {}
    stats = Counter()
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        parts = _split_docs(text)
        kept = 0
        for p in parts:
            ok, reason = _quality_ok(
                p,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                min_printable=args.min_printable,
                max_word_repeat=args.max_word_repeat,
            )
            stats[reason] += 1
            if not ok:
                continue
            if args.max_chars > 0 and len(p) > args.max_chars:
                p = p[: args.max_chars]
            docs.append(p)
            kept += 1
        src_stats[path.name] = kept
        print(f"source {path.name}: kept {kept:,} / {len(parts):,} docs")

    # exact dedup
    seen: set[str] = set()
    uniq: list[str] = []
    for d in docs:
        h = _norm_hash(d)
        if h in seen:
            stats["dedup"] += 1
            continue
        seen.add(h)
        uniq.append(d)
    docs = uniq
    print(f"exact-unique docs={len(docs):,}")

    # MinHash near-dedup (semantic-ish paraphrase / near-copy collapse)
    if args.near_dedup_jaccard > 0:
        docs, n_near = _near_dedup(docs, jaccard_thresh=args.near_dedup_jaccard)
        stats["near_dedup"] = n_near
        print(f"near-dedup dropped={n_near:,} remain={len(docs):,}")

    # Density bias: short atomic cards first in shuffle weights
    if args.prefer_short_weight > 1.0 and docs:
        weights = []
        for d in docs:
            # shorter → higher weight
            w = args.prefer_short_weight if len(d) < 400 else (1.2 if len(d) < 800 else 1.0)
            weights.append(w)
        # weighted shuffle via order keys
        order = list(range(len(docs)))
        order.sort(key=lambda i: rng.random() ** (1.0 / weights[i]))
        docs = [docs[i] for i in order]
    else:
        rng.shuffle(docs)

    n_val = max(1, int(len(docs) * args.val_ratio))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    print(f"docs train={len(train_docs):,} val={len(val_docs):,}")

    train_text = "\n\n".join(train_docs)
    val_text = "\n\n".join(val_docs)
    (out / "mix_train.txt").write_text(train_text, encoding="utf-8")
    (out / "mix_val.txt").write_text(val_text, encoding="utf-8")

    print(f"BPE train sample ≤{args.bpe_train_chars:,} chars...")
    bpe_text = _stratified_sample(train_docs, args.bpe_train_chars, rng)
    tok = train_bpe(bpe_text, vocab_size=args.vocab_size)
    tok_path = out / "tokenizer.json"
    tok.save(tok_path)
    print(f"tokenizer → {tok_path} vocab={tok.vocab_size}")

    print("encoding train/val bins...")
    n_train = _encode(tok, train_text, out / "mix_train.bin")
    n_val = _encode(tok, val_text, out / "mix_val.bin")
    print(f"tokens train={n_train:,} val={n_val:,}")

    lengths = [len(d) for d in train_docs[:50_000]] or [0]
    manifest = {
        "goal": "base_english",
        "raw_dir": str(raw),
        "sources": src_stats,
        "quality_stats": dict(stats),
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": n_train,
        "val_tokens": n_val,
        "vocab_size": tok.vocab_size,
        "shakespeare_ratio": 0.0,
        "seed": args.seed,
        "train_story_chars_mean": statistics.mean(lengths),
        "train_story_chars_median": statistics.median(lengths),
        "chars_per_tok_train": len(train_text) / max(n_train, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("prepare_base_english done →", out)


if __name__ == "__main__":
    main()
