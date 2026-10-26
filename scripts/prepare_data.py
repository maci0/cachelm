#!/usr/bin/env python3
"""Build mixed train corpus + small domain BPE tokenizer for ~75M L3 train.

Default mix: TinyStoriesV2-GPT4 (narrative English) + light Tiny Shakespeare.

Pipeline:
  - Prefer GPT-4 TinyStories when present (higher quality than GPT-3.5 mix)
  - Story-level train/val split (no char-cut leakage across stories)
  - Drop incomplete last story on byte-range partial downloads
  - Near-exact dedup (whitespace-normalized hash)
  - Train/val cross-dedup (no official-val stories in train)
  - Quality filters: length, printable ratio, repetition, sentence-ish end
  - Stratified BPE train sample (short/mid/long stories)
  - Official val file when present; pure stories in val (no Shakespeare)

Writes:
  data/mix_train.txt, data/mix_val.txt
  data/tokenizer.json
  data/mix_train.bin, data/mix_val.bin
  data/mix_manifest.json
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

EOT = "<|endoftext|>"
# TinyStories HF files use this delimiter between stories.
_SPLIT_RE = re.compile(r"(?:<\|endoftext\|>|\n{2,})")
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_END_RE = re.compile(r'[.!?]"?\s*$')


def _split_stories(text: str, *, drop_incomplete_tail: bool = True) -> list[str]:
    """Split corpus into story-sized chunks.

    Partial range downloads often cut mid-story. If the file does not end on a
    clean boundary (EOT or double newline), drop the last fragment.
    """
    ends_clean = bool(re.search(r"(?:<\|endoftext\|>|\n{2,})\s*$", text))
    parts = _SPLIT_RE.split(text)
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        s = s.replace(EOT, "").strip()
        if s:
            out.append(s)
    if drop_incomplete_tail and out and not ends_clean:
        # last piece is almost certainly truncated mid-story
        out.pop()
    return out


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _story_hash(s: str) -> str:
    return hashlib.sha1(_norm_key(s).encode("utf-8")).hexdigest()


def _dedup(stories: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    out: list[str] = []
    dropped = 0
    for s in stories:
        h = _story_hash(s)
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        out.append(s)
    return out, dropped


def _quality_ok(
    s: str,
    *,
    min_chars: int,
    max_chars: int,
    min_printable: float,
    max_word_repeat: float,
    require_sentence_end: bool,
) -> tuple[bool, str]:
    """Return (keep, reason). reason is empty when kept."""
    n = len(s)
    if n < min_chars:
        return False, "short"
    if max_chars > 0 and n > max_chars:
        return False, "long"  # caller may trim first; hard reject if still long
    # printable / letter density (drop binary junk / OCR garbage)
    printable = sum(1 for c in s if c.isprintable() or c in "\n\t")
    if printable / n < min_printable:
        return False, "nonprintable"
    letters = sum(1 for c in s if c.isalpha())
    if letters / n < 0.55:
        return False, "low_alpha"
    words = _WORD_RE.findall(s)
    if len(words) < 12:
        return False, "few_words"
    # dominant-word share (catch "the the the the..." spam)
    counts = Counter(w.lower() for w in words)
    top = counts.most_common(1)[0][1]
    if top / len(words) > max_word_repeat:
        return False, "repeat"
    # crude bigram repetition: same consecutive word pair density
    if len(words) >= 20:
        pairs = list(zip(words, words[1:]))
        pc = Counter((a.lower(), b.lower()) for a, b in pairs)
        if pc.most_common(1)[0][1] / len(pairs) > 0.12:
            return False, "bigram_repeat"
    if require_sentence_end and not _SENT_END_RE.search(s.rstrip()):
        return False, "no_end"
    return True, ""


def _trim_long(s: str, max_chars: int) -> str:
    """Keep story start up to max_chars on a word boundary."""
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    cut = s[:max_chars].rsplit(" ", 1)[0].strip()
    return cut


def _load_stories(
    paths: list[Path],
    min_chars: int,
    max_chars: int = 0,
    *,
    min_printable: float = 0.98,
    max_word_repeat: float = 0.18,
    require_sentence_end: bool = True,
    drop_incomplete_tail: bool = True,
) -> tuple[list[str], dict[str, int]]:
    all_s: list[str] = []
    totals: Counter[str] = Counter()
    for p in paths:
        if not p.is_file():
            print(f"skip missing source: {p}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        stories = _split_stories(text, drop_incomplete_tail=drop_incomplete_tail)
        before = len(stories)
        kept: list[str] = []
        stats: Counter[str] = Counter()
        for s in stories:
            if max_chars > 0 and len(s) > max_chars:
                s = _trim_long(s, max_chars)
                stats["long_trim"] += 1
            ok, reason = _quality_ok(
                s,
                min_chars=min_chars,
                max_chars=max_chars,
                min_printable=min_printable,
                max_word_repeat=max_word_repeat,
                require_sentence_end=require_sentence_end,
            )
            if not ok:
                stats[reason] += 1
                continue
            kept.append(s)
            stats["kept"] += 1
        drop_bits = ", ".join(
            f"{k}:{v}" for k, v in sorted(stats.items()) if k != "kept"
        )
        print(
            f"  {p.name}: {before:,} stories → {len(kept):,} "
            f"(min={min_chars}, max={max_chars or '∞'}, drops={{{drop_bits}}})"
        )
        all_s.extend(kept)
        totals.update(stats)
    return all_s, dict(totals)


def _length_hist(stories: list[str]) -> dict[str, int]:
    bins = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 10**9)]
    labels = ["0-200", "200-500", "500-1k", "1k-2k", "2k-4k", "4k+"]
    hist = {lab: 0 for lab in labels}
    for s in stories:
        n = len(s)
        for (lo, hi), lab in zip(bins, labels):
            if lo <= n < hi:
                hist[lab] += 1
                break
    return hist


def _stratified_bpe_sample(
    stories: list[str], max_chars: int, rng: random.Random
) -> str:
    """Build a length-stratified text sample for BPE training."""
    if max_chars <= 0 or not stories:
        return "\n\n".join(stories)
    # three strata by char length
    short = [s for s in stories if len(s) < 500]
    mid = [s for s in stories if 500 <= len(s) < 1500]
    long = [s for s in stories if len(s) >= 1500]
    for bucket in (short, mid, long):
        rng.shuffle(bucket)
    # target share: 25% short / 50% mid / 25% long (mid is TinyStories bulk)
    budgets = {
        "short": int(max_chars * 0.25),
        "mid": int(max_chars * 0.50),
        "long": int(max_chars * 0.25),
    }
    pieces: list[str] = []
    used = 0
    for name, bucket in (("short", short), ("mid", mid), ("long", long)):
        budget = budgets[name]
        take = 0
        for s in bucket:
            if take >= budget:
                break
            pieces.append(s)
            take += len(s) + 2
            used += len(s) + 2
    # fill remainder from any remaining stories if under budget
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
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _encode_file(tok, text: str, out_bin: Path) -> int:
    ids = tok.encode(text)
    arr = np.asarray(ids, dtype=np.uint16 if tok.vocab_size < 65535 else np.int32)
    arr.tofile(out_bin)
    return len(arr)


def _join_stories(stories: list[str], sep: str = "\n\n") -> str:
    return sep.join(stories)


def _resolve_story_paths(out: Path, root: Path, explicit: list[str] | None) -> list[Path]:
    if explicit is not None:
        paths: list[Path] = []
        for s in explicit:
            p = Path(s)
            if not p.is_file():
                p = root / s
            paths.append(p)
        return paths
    # Prefer single best corpus: full GPT-4 > partial GPT-4 > older mixed
    candidates = [
        out / "TinyStoriesV2-GPT4-train.txt",
        out / "TinyStoriesV2-GPT4-train.partial.txt",
        out / "TinyStories-train.txt",
        out / "TinyStories-train.partial.txt",
        root / "data" / "TinyStoriesV2-GPT4-train.txt",
        root / "data" / "TinyStoriesV2-GPT4-train.partial.txt",
        root / "data" / "TinyStories-train.partial.txt",
    ]
    for c in candidates:
        if c.is_file():
            return [c]
    raise SystemExit("no TinyStories file found; run scripts/download_data.sh first")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stories",
        type=str,
        nargs="+",
        default=None,
        help="one or more TinyStories-like text files (auto-detect if omitted)",
    )
    ap.add_argument("--shakespeare", type=str, default="data/tinyshakespeare.txt")
    ap.add_argument(
        "--val-stories",
        type=str,
        default=None,
        help="optional official val file (story-level; not mixed into train)",
    )
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument(
        "--shakespeare-ratio",
        type=float,
        default=0.05,
        help="char fraction of train mix from Shakespeare (rest = stories; default 0.05)",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="story fraction held out when no --val-stories / official val",
    )
    ap.add_argument("--min-chars", type=int, default=120, help="drop shorter stories")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="truncate stories longer than this (0 = no max)",
    )
    ap.add_argument(
        "--max-train-chars",
        type=int,
        default=0,
        help="cap total train chars after mix (0 = no cap)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--bpe-train-chars",
        type=int,
        default=40_000_000,
        help="chars used to train BPE (stratified sample; more = better domain merges)",
    )
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument(
        "--no-quality",
        action="store_true",
        help="disable quality filters beyond length",
    )
    ap.add_argument(
        "--interleave-chunk",
        type=int,
        default=4,
        help="stories per chunk when interleaving shakespeare",
    )
    ap.add_argument(
        "--min-printable",
        type=float,
        default=0.98,
        help="min fraction of printable chars",
    )
    ap.add_argument(
        "--max-word-repeat",
        type=float,
        default=0.18,
        help="max share of a single word in a story",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    story_paths = _resolve_story_paths(out, root, args.stories)
    shake_p = Path(args.shakespeare)
    if not shake_p.is_file():
        shake_p = root / args.shakespeare

    q_kwargs = dict(
        min_printable=args.min_printable,
        max_word_repeat=args.max_word_repeat,
        require_sentence_end=not args.no_quality,
        drop_incomplete_tail=True,
    )
    if args.no_quality:
        q_kwargs["min_printable"] = 0.0
        q_kwargs["max_word_repeat"] = 1.0
        q_kwargs["require_sentence_end"] = False

    print("loading stories:")
    stories, load_stats = _load_stories(
        story_paths, args.min_chars, args.max_chars, **q_kwargs
    )
    if not args.no_dedup:
        stories, n_dup = _dedup(stories)
        print(f"dedup: dropped {n_dup:,} near-exact; {len(stories):,} unique remain")
    else:
        n_dup = 0
        print(f"no dedup: {len(stories):,} stories")

    rng = random.Random(args.seed)
    rng.shuffle(stories)

    # Val set: official file or story-level holdout
    val_stories: list[str] = []
    val_source = "holdout"
    if args.val_stories:
        vp = Path(args.val_stories)
        if not vp.is_file():
            vp = root / args.val_stories
        val_stories, _ = _load_stories([vp], args.min_chars, args.max_chars, **q_kwargs)
        if not args.no_dedup:
            val_stories, _ = _dedup(val_stories)
        train_stories = stories
        val_source = str(vp)
    else:
        auto_val = out / "TinyStoriesV2-GPT4-valid.txt"
        if not auto_val.is_file():
            auto_val = root / "data" / "TinyStoriesV2-GPT4-valid.txt"
        if auto_val.is_file():
            print(f"using official val: {auto_val.name}")
            val_stories, _ = _load_stories(
                [auto_val], args.min_chars, args.max_chars, **q_kwargs
            )
            if not args.no_dedup:
                val_stories, _ = _dedup(val_stories)
            train_stories = stories
            val_source = str(auto_val)
        else:
            n_val = max(1, int(len(stories) * args.val_ratio))
            val_stories = stories[:n_val]
            train_stories = stories[n_val:]
            val_source = f"holdout:{args.val_ratio}"

    # Cross-dedup: drop any train story that appears in val (leakage)
    n_leak = 0
    if not args.no_dedup and val_stories:
        val_keys = {_story_hash(s) for s in val_stories}
        before = len(train_stories)
        train_stories = [s for s in train_stories if _story_hash(s) not in val_keys]
        n_leak = before - len(train_stories)
        if n_leak:
            print(f"train/val cross-dedup: removed {n_leak:,} leaked train stories")

    print(f"train stories: {len(train_stories):,}  val stories: {len(val_stories):,}")
    train_hist = _length_hist(train_stories)
    print(f"train length hist: {train_hist}")

    # Shakespeare as style spice (character-budget relative to story chars)
    shake = (
        shake_p.read_text(encoding="utf-8", errors="replace") if shake_p.is_file() else ""
    )
    story_train_txt = _join_stories(train_stories)
    s_chars = len(story_train_txt)
    k_budget = int(
        s_chars * args.shakespeare_ratio / max(1.0 - args.shakespeare_ratio, 1e-6)
    )
    shake_chars = 0
    if shake and k_budget > 0:
        k_part = (shake * (k_budget // max(len(shake), 1) + 1))[:k_budget]
        shake_chars = len(k_part)
        shake_paras = [p.strip() for p in re.split(r"\n{2,}", k_part) if p.strip()]
        rng.shuffle(shake_paras)
        pieces: list[str] = []
        si = 0
        for i in range(0, len(train_stories), args.interleave_chunk):
            chunk = train_stories[i : i + args.interleave_chunk]
            pieces.append(_join_stories(chunk))
            if si < len(shake_paras):
                pieces.append(shake_paras[si])
                si += 1
        while si < len(shake_paras):
            pieces.append(shake_paras[si])
            si += 1
        rng.shuffle(pieces)
        train_txt = "\n\n".join(pieces)
        print(
            f"shakespeare: {shake_chars:,} chars "
            f"({100 * shake_chars / max(len(train_txt), 1):.1f}% of train)"
        )
    else:
        train_txt = story_train_txt
        print("shakespeare: none")

    if args.max_train_chars > 0 and len(train_txt) > args.max_train_chars:
        train_txt = train_txt[: args.max_train_chars]
        print(f"capped train to {args.max_train_chars:,} chars")

    val_txt = _join_stories(val_stories)
    # keep val pure stories (no shakespeare) for cleaner NLL signal
    (out / "mix_train.txt").write_text(train_txt, encoding="utf-8")
    (out / "mix_val.txt").write_text(val_txt, encoding="utf-8")
    print(f"wrote mix_train {len(train_txt):,} chars  mix_val {len(val_txt):,} chars")

    bpe_src = _stratified_bpe_sample(train_stories, args.bpe_train_chars, rng)
    print(
        f"training BPE vocab_size={args.vocab_size} on "
        f"{len(bpe_src):,} stratified chars..."
    )
    tok = train_bpe(bpe_src, vocab_size=args.vocab_size, max_chars=len(bpe_src))
    tok_path = out / "tokenizer.json"
    tok.save(tok_path)
    sample = "Once upon a time, Lily found a needle.\nROMEO:\n"
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"tokenizer {tok.vocab_size} tokens → {tok_path}")
    print(f"encode sample: {len(ids)} ids  roundtrip ok={back[:40]!r}...")

    print("pre-encoding train/val to binary token files...")
    ntr = _encode_file(tok, train_txt, out / "mix_train.bin")
    nva = _encode_file(tok, val_txt, out / "mix_val.bin")
    print(f"wrote mix_train.bin ({ntr:,} tok)  mix_val.bin ({nva:,} tok)")
    cpt_tr = len(train_txt) / max(ntr, 1)
    cpt_va = len(val_txt) / max(nva, 1)
    print(f"avg train chars/tok: {cpt_tr:.2f}  val: {cpt_va:.2f}")

    lens = [len(s) for s in train_stories] or [0]
    manifest = {
        "target_model": "75M L27 d512 h8/kv1",
        "story_sources": [str(p) for p in story_paths],
        "story_source_bytes": [p.stat().st_size if p.is_file() else 0 for p in story_paths],
        "shakespeare": str(shake_p) if shake_p.is_file() else None,
        "val_source": val_source,
        "vocab_size": tok.vocab_size,
        "shakespeare_ratio": args.shakespeare_ratio,
        "shakespeare_chars": shake_chars,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "bpe_train_chars": args.bpe_train_chars,
        "bpe_sample_chars": len(bpe_src),
        "seed": args.seed,
        "quality": {
            "enabled": not args.no_quality,
            "min_printable": args.min_printable,
            "max_word_repeat": args.max_word_repeat,
            "require_sentence_end": not args.no_quality,
            "load_stats": load_stats,
            "dedup_dropped": n_dup,
            "train_val_leak_removed": n_leak,
        },
        "train_stories": len(train_stories),
        "val_stories": len(val_stories),
        "train_chars": len(train_txt),
        "val_chars": len(val_txt),
        "train_tokens": ntr,
        "val_tokens": nva,
        "chars_per_tok_train": cpt_tr,
        "chars_per_tok_val": cpt_va,
        "train_length_hist": train_hist,
        "train_story_chars_mean": statistics.fmean(lens),
        "train_story_chars_median": statistics.median(lens),
    }
    (out / "mix_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out / 'mix_manifest.json'}")
    # rough chinchilla note for 75M
    chinchilla = 75_000_000 * 20
    epochs_eq = ntr / chinchilla
    print(
        f"tokens vs Chinchilla@75M (~{chinchilla:,} tok): "
        f"{ntr:,} train tok ≈ {epochs_eq:.2f}× one optimal pass "
        f"(multi-epoch fine on TinyStories)"
    )


if __name__ == "__main__":
    main()
