"""Small domain BPE tokenizer (pure Python, no external deps).

Trains a GPT-2-style byte-level BPE with a modest vocab (default 2048) so
embedding tables stay L3-friendly while sequences pack more English per token
than raw characters.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_SPLIT_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def _pretokenize(text: str) -> list[str]:
    return _SPLIT_RE.findall(text)


def _word_to_symbols(word: str) -> tuple[str, ...]:
    # UTF-8 bytes as single-char symbols (latin-1 code units 0-255)
    return tuple(chr(b) for b in word.encode("utf-8"))


@dataclass
class BPETokenizer:
    """Encode/decode with merges list and id maps."""

    stoi: dict[str, int]
    itos: dict[int, str]
    merges: list[tuple[str, str]]  # in merge order
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        """Encode text. Uses rank table for O(n) BPE merges per word."""
        ids: list[int] = []
        unk = self.stoi.get("<unk>", 0)
        # merge ranks: lower = earlier merge (higher priority)
        if not hasattr(self, "_ranks") or self._ranks is None:
            self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        ranks = self._ranks
        for word in _pretokenize(text):
            syms = list(_word_to_symbols(word))
            if not syms:
                continue
            if not ranks:
                for s in syms:
                    ids.append(self.stoi.get(s, unk))
                continue
            # GPT-2 style: repeatedly merge best pair by rank
            while len(syms) >= 2:
                best_i = -1
                best_rank = 10**18
                for i in range(len(syms) - 1):
                    r = ranks.get((syms[i], syms[i + 1]))
                    if r is not None and r < best_rank:
                        best_rank = r
                        best_i = i
                if best_i < 0:
                    break
                a, b = syms[best_i], syms[best_i + 1]
                syms = syms[:best_i] + [a + b] + syms[best_i + 2 :]
            for s in syms:
                ids.append(self.stoi.get(s, unk))
        return ids

    def decode(self, ids: list[int] | "torch.Tensor") -> str:
        if hasattr(ids, "tolist"):
            ids = ids.view(-1).tolist()  # type: ignore[assignment]
        parts: list[str] = []
        for i in ids:
            try:
                ii = int(i)
            except (TypeError, ValueError):
                continue
            # Out-of-range / unknown / special ids: skip (never raise)
            if ii < 0:
                continue
            s = self.itos.get(ii, "")
            if not s or s in ("<unk>", "<pad>", "<bos>", "<eos>"):
                continue
            parts.append(s)
        raw = "".join(parts)
        try:
            return raw.encode("latin-1").decode("utf-8", errors="replace")
        except Exception:
            return raw

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(
                {
                    "vocab_size": self.vocab_size,
                    "stoi": self.stoi,
                    "merges": [list(m) for m in self.merges],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "BPETokenizer":
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        if "stoi" not in obj or "merges" not in obj:
            raise ValueError(f"tokenizer file missing stoi/merges: {path}")
        stoi = {str(k): int(v) for k, v in obj["stoi"].items()}
        if not stoi:
            raise ValueError(f"tokenizer stoi empty: {path}")
        # Last-wins on duplicate ids would drop tokens silently; prefer lowest id
        itos: dict[int, str] = {}
        for s, i in stoi.items():
            if i not in itos:
                itos[i] = s
        merges: list[tuple[str, str]] = []
        for m in obj["merges"]:
            if not isinstance(m, (list, tuple)) or len(m) != 2:
                raise ValueError(f"bad merge entry in {path}: {m!r}")
            merges.append((str(m[0]), str(m[1])))
        vs = int(obj.get("vocab_size", len(stoi)))
        if vs < 1:
            raise ValueError(f"tokenizer vocab_size invalid: {vs}")
        return cls(stoi=stoi, itos=itos, merges=merges, vocab_size=vs)


def train_bpe(
    text: str,
    vocab_size: int = 2048,
    max_chars: int = 8_000_000,
) -> BPETokenizer:
    """Train byte-level BPE. Special tokens occupy the first few ids."""
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
    # base: all bytes that appear + full 0-255 for robustness
    base_syms = [chr(i) for i in range(256)]
    stoi: dict[str, int] = {}
    for s in specials:
        stoi[s] = len(stoi)
    for s in base_syms:
        if s not in stoi:
            stoi[s] = len(stoi)

    # corpus as list of symbol lists (per pretoken word), with counts
    word_counts: Counter[tuple[str, ...]] = Counter()
    for word in _pretokenize(text):
        word_counts[_word_to_symbols(word)] += 1

    merges: list[tuple[str, str]] = []
    # work on mutable sequences of symbols
    corpus: list[list[str]] = [list(w) for w, c in word_counts.items() for _ in range(1)]
    # weight by count via parallel weights
    weights = [c for _, c in word_counts.items()]
    sequences = [list(w) for w, _ in word_counts.items()]

    def pair_stats() -> Counter[tuple[str, str]]:
        stats: Counter[tuple[str, str]] = Counter()
        for seq, w in zip(sequences, weights):
            for i in range(len(seq) - 1):
                stats[(seq[i], seq[i + 1])] += w
        return stats

    def merge_pair(a: str, b: str) -> None:
        ab = a + b
        for si, seq in enumerate(sequences):
            i = 0
            out: list[str] = []
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(ab)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            sequences[si] = out

    target_merges = max(0, vocab_size - len(stoi))
    for _ in range(target_merges):
        stats = pair_stats()
        if not stats:
            break
        (a, b), _ = stats.most_common(1)[0]
        merge_pair(a, b)
        merges.append((a, b))
        ab = a + b
        if ab not in stoi:
            stoi[ab] = len(stoi)
        if len(stoi) >= vocab_size:
            break

    # trim if overshot
    if len(stoi) > vocab_size:
        # keep first vocab_size by id
        stoi = {s: i for s, i in stoi.items() if i < vocab_size}
        merges = merges[: max(0, vocab_size - 256 - len(specials))]

    itos = {i: s for s, i in stoi.items()}
    return BPETokenizer(stoi=stoi, itos=itos, merges=merges, vocab_size=len(stoi))

