#!/usr/bin/env python3
"""Download / synthesize base-English sources (docs/BASE_ENGLISH.md).

Writes plain text under data/base_en/raw/ (or --out-dir). Prefer running on
spark2 (disk); local home is often tight.

Sources:
  - HuggingFaceFW/fineweb-edu (streaming educational web)
  - synthetic short explainers + dialogue (local templates, no API)
  - optional TinyStories spice (default 0 chars)

  python scripts/download_base_english.py --fineweb-chars 1500000000
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path


def _write_capped(path: Path, lines: list[str], max_chars: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            if n >= max_chars:
                break
            chunk = line.rstrip() + "\n\n"
            f.write(chunk)
            n += len(chunk)
    return n



def gen_synthetic(rng: random.Random, max_chars: int) -> list[str]:
    """Compressed KB cards + basic English (no tools, no TinyStories voice).

    Loads ll3m/kb_facts.json when present for high knowledge density.
    Fills up to max_chars by looping templates.
    """
    import json
    from pathlib import Path as _P

    kb_path = _P(__file__).resolve().parent.parent / "ll3m" / "kb_facts.json"
    kb: dict[str, str] = {}
    if kb_path.is_file():
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and (raw.get("version") == 2 or "flat" in raw):
            flat = raw.get("flat") or {}
            if flat:
                kb = {str(k): str(v) for k, v in flat.items()}
            else:
                for ent in raw.get("entries") or []:
                    val = str(ent.get("value", ""))
                    eid = str(ent.get("id", ""))
                    if eid:
                        kb[eid] = val
                    for a in ent.get("aliases") or []:
                        kb[str(a)] = val
        else:
            skip = {"version", "note", "n_entries", "n_keys", "entries", "flat"}
            kb = {
                str(k): str(v)
                for k, v in raw.items()
                if k not in skip and not isinstance(v, (dict, list))
            }
    kb_items = list(kb.items()) or [("pi", "3.14159"), ("water formula", "H2O")]

    topics = [
        ("gravity", "Gravity pulls objects toward each other. On Earth, it makes things fall down."),
        ("water cycle", "Water evaporates, forms clouds, then falls as rain or snow."),
        ("photosynthesis", "Plants use sunlight, water, and carbon dioxide to make food and oxygen."),
        ("democracy", "In a democracy, people choose leaders through free elections."),
        ("electricity", "Electric current is the flow of charge through a conductor."),
        ("fractions", "A fraction shows a part of a whole, like 1/2 of a cake."),
        ("algorithm", "An algorithm is a step-by-step plan for solving a problem."),
        ("atom", "An atom is the basic unit of matter."),
        ("evolution", "Evolution is change in living things over many generations."),
        ("ecosystem", "An ecosystem is a community of living things and their environment."),
    ]
    out: list[str] = []
    total = 0

    def add(s: str) -> bool:
        nonlocal total
        if total >= max_chars:
            return False
        out.append(s)
        total += len(s) + 2
        return True

    while total < max_chars:
        # Bias hard toward atomic facts (knowledge density); light prose for English only.
        kind = rng.choices(
            ["kb_qa", "kb_fact", "kb_card", "explainer", "dialogue", "math", "prose"],
            weights=[34, 24, 18, 8, 6, 8, 2],
            k=1,
        )[0]
        if kind in ("kb_qa", "kb_fact", "kb_card"):
            k, v = rng.choice(kb_items)
            if kind == "kb_qa":
                s = f"User: What is {k}?\nAssistant: {v}."
            elif kind == "kb_fact":
                s = f"Fact: {k} = {v}."
            else:
                topic, fact = rng.choice(topics)
                s = (
                    f"Knowledge card.\n"
                    f"Topic: {topic}. {fact}\n"
                    f"Fact: {k} is {v}."
                )
        elif kind == "explainer":
            topic, fact = rng.choice(topics)
            s = f"{fact} This is a simple explanation of {topic}."
        elif kind == "dialogue":
            topic, fact = rng.choice(topics)
            s = f"User: Can you explain {topic} simply?\nAssistant: Yes. {fact}"
        elif kind == "math":
            a, b = rng.randint(2, 40), rng.randint(2, 40)
            op = rng.choice(["plus", "minus", "times"])
            if op == "plus":
                ans = a + b
            elif op == "minus":
                a, b = max(a, b), min(a, b)
                ans = a - b
            else:
                ans = a * b
            s = f"User: What is {a} {op} {b}?\nAssistant: {a} {op} {b} is {ans}."
        else:
            openers = [
                "In everyday life,",
                "A useful rule is that",
                "When learning something new,",
                "Clear writing matters because",
            ]
            bodies = [
                "people solve problems by breaking them into smaller steps.",
                "practice improves skill more than talent alone.",
                "simple language often communicates better than complex jargon.",
                "asking a precise question leads to a more useful answer.",
            ]
            s = f"{rng.choice(openers)} {rng.choice(bodies)} {rng.choice(bodies)}"
        if not add(s):
            break
    return out


def stream_fineweb_edu(out_path: Path, max_chars: int, seed: int) -> int:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "need datasets: uv pip install --python .venv/bin/python datasets huggingface_hub"
        ) from e

    # Prefer project cache if HF_HOME set by pipeline (avoids root-owned ~/.cache)
    import os

    print(
        f"HF_HOME={os.environ.get('HF_HOME', '')!r} "
        f"hub={os.environ.get('HUGGINGFACE_HUB_CACHE', '')!r}"
    )

    # sample-10BT is a manageable edu subset; streaming avoids full download
    print(f"streaming HuggingFaceFW/fineweb-edu sample-10BT → {out_path} (cap {max_chars:,} chars)")
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
    except Exception as e1:
        print(f"sample-10BT failed ({e1}); trying default config streaming...", flush=True)
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_docs = 0
    # Prefer shorter edu chunks (higher fact density per char than long pages).
    min_len = 120
    max_doc = 1200
    printable_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            if n >= max_chars:
                break
            text = row.get("text") or ""
            if not isinstance(text, str):
                continue
            text = printable_re.sub("", text).strip()
            if len(text) < min_len:
                continue
            # Prefer higher edu scores when present
            score = row.get("score")
            if score is not None:
                try:
                    if float(score) < 1.5:
                        continue
                except (TypeError, ValueError):
                    pass
            if len(text) > max_doc:
                text = text[:max_doc].rsplit(" ", 1)[0]
            chunk = text + "\n\n"
            f.write(chunk)
            n += len(chunk)
            n_docs += 1
            if n_docs % 500 == 0:
                print(f"  fineweb docs={n_docs:,} chars={n:,}", flush=True)

    print(f"fineweb done: {n_docs:,} docs, {n:,} chars → {out_path}")
    return n


def copy_stories_spice(src: Path, out_path: Path, max_chars: int) -> int:
    if max_chars <= 0 or not src.is_file():
        return 0
    text = src.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"stories spice: {len(text):,} chars from {src}")
    return len(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str, default="data/base_en/raw")
    ap.add_argument("--fineweb-chars", type=int, default=1_500_000_000)
    ap.add_argument("--synthetic-chars", type=int, default=250_000_000)
    ap.add_argument("--stories-chars", type=int, default=0, help="optional TinyStories spice")
    ap.add_argument(
        "--stories-src",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.partial.txt",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-fineweb", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    syn_path = out / "synthetic.txt"
    if args.synthetic_chars > 0:
        print(f"generating synthetic ≤{args.synthetic_chars:,} chars (streaming)...")
        # Stream write to avoid holding hundreds of MB as a Python list
        syn_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        # gen_synthetic still builds a list; for large budgets, write in chunks
        chunk_budget = min(args.synthetic_chars, 20_000_000)
        with syn_path.open("w", encoding="utf-8") as f:
            while n < args.synthetic_chars:
                need = min(chunk_budget, args.synthetic_chars - n)
                docs = gen_synthetic(rng, need)
                for line in docs:
                    if n >= args.synthetic_chars:
                        break
                    chunk = line.rstrip() + "\n\n"
                    f.write(chunk)
                    n += len(chunk)
                if not docs:
                    break
                print(f"  synthetic chars={n:,}", flush=True)
        print(f"synthetic: {n:,} chars → {syn_path}")

    fw_path = out / "fineweb_edu.txt"
    if not args.skip_fineweb and args.fineweb_chars > 0:
        stream_fineweb_edu(fw_path, args.fineweb_chars, args.seed)
    elif args.skip_fineweb:
        print("skip fineweb")

    if args.stories_chars > 0:
        copy_stories_spice(Path(args.stories_src), out / "stories_spice.txt", args.stories_chars)

    print("download_base_english done:", out)


if __name__ == "__main__":
    main()
