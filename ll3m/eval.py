"""
Capability evaluation for ll3m.

A cache-sized char-LM is not an MMLU model. We measure what it *can* be good at:

  1. Compression quality   — BPC / NLL on held-out Shakespeare (primary scalar)
  2. Domain discrimination — does Shakespeare compress better than foreign text?
  3. Next-char ranking     — top-1 / top-5 accuracy on val
  4. Line completion       — teacher-forced NLL + greedy recovery of val suffixes
  5. Format competence     — do free samples look like the corpus (speakers, lines)?
  6. Diversity             — unique char/word n-grams (collapse detector)
  7. Cache claim           — tensor working set vs L3 budget

Usage:
  uv run python -m ll3m.eval --checkpoint checkpoints/best.pt
  uv run python -m ll3m.eval --checkpoint checkpoints/best.pt --json out/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ll3m.cache_info import probe_cache
from ll3m.config import GPTConfig
from ll3m.model_torch import estimate_budget
from ll3m.train import decode, load_model, load_text


# ---------------------------------------------------------------------------
# Metrics containers
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    checkpoint: str
    step: int | None
    config: dict
    n_params: int
    # 1. compression
    val_nll: float
    val_bpc: float
    val_ppl: float
    train_nll: float | None = None
    # 2. domain
    domain: dict = field(default_factory=dict)
    # 3. ranking
    top1_acc: float = 0.0
    top5_acc: float = 0.0
    # 4. completion
    completion: dict = field(default_factory=dict)
    # 5. format
    format_score: dict = field(default_factory=dict)
    # 6. diversity
    diversity: dict = field(default_factory=dict)
    # 7. cache
    cache: dict = field(default_factory=dict)
    # samples
    samples: list[str] = field(default_factory=list)
    # wall
    eval_seconds: float = 0.0
    # single scalar for leaderboards (lower is better): val BPC
    primary_score_bpc: float = 0.0
    # higher is better composite in [0, 1]
    capability_index: float = 0.0


# ---------------------------------------------------------------------------
# Core NLL
# ---------------------------------------------------------------------------


@torch.no_grad()
def nll_over_data(
    model,
    data: torch.Tensor,
    block_size: int,
    batch_size: int = 32,
    max_batches: int | None = None,
) -> tuple[float, int]:
    """Mean token NLL over sliding windows into `data` (contiguous, no drop)."""
    model.eval()
    device = next(model.parameters()).device
    n = len(data)
    # non-overlapping blocks for unbiased coverage
    starts = list(range(0, n - block_size - 1, block_size))
    if not starts:
        return float("nan"), 0
    total_loss = 0.0
    total_tok = 0
    n_batches = 0
    for i in range(0, len(starts), batch_size):
        if max_batches is not None and n_batches >= max_batches:
            break
        chunk = starts[i : i + batch_size]
        x = torch.stack([data[s : s + block_size] for s in chunk]).to(device)
        y = torch.stack([data[s + 1 : s + block_size + 1] for s in chunk]).to(device)
        _, loss = model(x, y)
        # loss is mean over batch*T; weight by token count
        tok = x.numel()
        total_loss += loss.item() * tok
        total_tok += tok
        n_batches += 1
    return total_loss / max(total_tok, 1), total_tok


@torch.no_grad()
def nll_of_text(model, text: str, stoi: dict[str, int], block_size: int) -> float:
    """NLL of a string under the model (unknown chars mapped to 0)."""
    if len(text) < 2:
        return float("nan")
    ids = torch.tensor(
        [stoi.get(c, 0) for c in text], dtype=torch.long
    )
    # chunk into block_size windows with stride block_size//2
    losses = []
    stride = max(1, block_size // 2)
    device = next(model.parameters()).device
    for s in range(0, len(ids) - 1, stride):
        chunk = ids[s : s + block_size + 1]
        if len(chunk) < 2:
            break
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)
        # pad to at least something model accepts — use actual length via crop
        if x.size(1) > block_size:
            x = x[:, -block_size:]
            y = y[:, -block_size:]
        _, loss = model(x, y)
        losses.append(loss.item())
        if s + block_size + 1 >= len(ids):
            break
    return float(sum(losses) / max(len(losses), 1))


# ---------------------------------------------------------------------------
# Top-k next-char accuracy
# ---------------------------------------------------------------------------


@torch.no_grad()
def topk_accuracy(
    model,
    data: torch.Tensor,
    block_size: int,
    batch_size: int = 32,
    max_batches: int = 80,
    ks: tuple[int, ...] = (1, 5),
) -> dict[int, float]:
    model.eval()
    device = next(model.parameters()).device
    starts = list(range(0, len(data) - block_size - 1, block_size))
    rng = np.random.default_rng(0)
    if len(starts) > max_batches * batch_size:
        starts = list(rng.choice(starts, size=max_batches * batch_size, replace=False))
    correct = {k: 0 for k in ks}
    total = 0
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        x = torch.stack([data[s : s + block_size] for s in chunk]).to(device)
        y = torch.stack([data[s + 1 : s + block_size + 1] for s in chunk]).to(device)
        logits, _ = model(x)
        # score every position
        for k in ks:
            top = logits.topk(k, dim=-1).indices  # B,T,k
            hit = (top == y.unsqueeze(-1)).any(dim=-1)
            correct[k] += int(hit.sum().item())
        total += y.numel()
    return {k: correct[k] / max(total, 1) for k in ks}


# ---------------------------------------------------------------------------
# Line completion (suffix recovery)
# ---------------------------------------------------------------------------


_SPEAKER_RE = re.compile(r"^[A-Z][A-Z0-9 '.-]{0,30}:\s*$")
_WORD_RE = re.compile(r"[A-Za-z']+")


@torch.no_grad()
def line_completion(
    model,
    text: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int,
    n_lines: int = 64,
    prefix_frac: float = 0.5,
) -> dict:
    """
    Take held-out lines (length >= 20), give the first prefix_frac as prompt,
    greedy-decode the rest, score exact match / char accuracy / true-suffix NLL.
    """
    lines = [
        ln for ln in text.splitlines()
        if 20 <= len(ln) <= 120 and not _SPEAKER_RE.match(ln)
    ]
    rng = np.random.default_rng(1)
    if len(lines) > n_lines:
        idx = rng.choice(len(lines), size=n_lines, replace=False)
        lines = [lines[i] for i in idx]

    device = next(model.parameters()).device
    model.eval()
    exact = 0
    char_hits = 0
    char_total = 0
    suffix_nlls = []

    for line in lines:
        cut = max(1, int(len(line) * prefix_frac))
        prefix, suffix = line[:cut], line[cut:]
        if not suffix:
            continue
        # NLL of true suffix given prefix
        full_ids = torch.tensor([stoi.get(c, 0) for c in line], dtype=torch.long, device=device)
        # teacher-forced NLL on suffix positions
        # build context window ending at each suffix char
        nll_sum = 0.0
        for j, ch in enumerate(suffix):
            pos = cut + j  # index in line of target char
            ctx = full_ids[max(0, pos - block_size) : pos]
            if ctx.numel() == 0:
                continue
            logits, _ = model(ctx.unsqueeze(0))
            logp = F.log_softmax(logits[0, -1], dim=-1)
            tid = stoi.get(ch, 0)
            nll_sum += -float(logp[tid])
        suffix_nlls.append(nll_sum / len(suffix))

        # greedy completion
        ids = torch.tensor(
            [[stoi.get(c, 0) for c in prefix]], dtype=torch.long, device=device
        )
        for _ in range(len(suffix)):
            cond = ids[:, -block_size:]
            logits, _ = model(cond)
            nxt = int(logits[0, -1].argmax())
            ids = torch.cat(
                [ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1
            )
        gen = decode(ids[0, len(prefix) :], itos)
        if gen == suffix:
            exact += 1
        for a, b in zip(gen, suffix):
            char_total += 1
            if a == b:
                char_hits += 1
        # if gen longer/shorter, only zip counts; pad miss
        if len(gen) < len(suffix):
            char_total += len(suffix) - len(gen)

    n = max(len(lines), 1)
    return {
        "n_lines": len(lines),
        "exact_match": exact / n,
        "char_accuracy": char_hits / max(char_total, 1),
        "suffix_nll": float(sum(suffix_nlls) / max(len(suffix_nlls), 1)),
        "suffix_bpc": float(sum(suffix_nlls) / max(len(suffix_nlls), 1) / math.log(2)),
    }


# ---------------------------------------------------------------------------
# Format competence on free samples
# ---------------------------------------------------------------------------


@torch.no_grad()
def format_and_diversity(
    model,
    stoi: dict[str, int],
    itos: dict[int, str],
    prompts: list[str],
    block_size: int,
    tokens_each: int = 400,
    temperature: float = 0.8,
    top_k: int = 40,
) -> tuple[dict, dict, list[str]]:
    device = next(model.parameters()).device
    model.eval()
    samples: list[str] = []
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    for prompt in prompts:
        ids = torch.tensor(
            [[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device
        )
        out = raw.generate(
            ids, max_new_tokens=tokens_each, temperature=temperature, top_k=top_k
        )
        samples.append(decode(out[0], itos))

    # format: speaker headers, avg line length, printable ratio, blank-line structure
    speaker_hits = 0
    total_lines = 0
    speaker_lines = 0
    line_lens = []
    printable = 0
    total_chars = 0
    for s in samples:
        lines = s.splitlines()
        total_lines += len(lines)
        for ln in lines:
            line_lens.append(len(ln))
            if _SPEAKER_RE.match(ln.strip()):
                speaker_lines += 1
        if any(_SPEAKER_RE.match(ln.strip()) for ln in lines):
            speaker_hits += 1
        for ch in s:
            total_chars += 1
            if ch.isprintable() or ch in "\n\t":
                printable += 1

    fmt = {
        "samples": len(samples),
        "frac_samples_with_speaker": speaker_hits / max(len(samples), 1),
        "frac_lines_speaker_header": speaker_lines / max(total_lines, 1),
        "mean_line_len": float(np.mean(line_lens)) if line_lens else 0.0,
        "median_line_len": float(np.median(line_lens)) if line_lens else 0.0,
        "printable_ratio": printable / max(total_chars, 1),
        # "dramatic" structure: not one giant line
        "lines_per_sample": total_lines / max(len(samples), 1),
    }

    # diversity: unique n-grams / total
    blob = "\n".join(samples)
    def unique_ratio(seq, n):
        if len(seq) < n:
            return 0.0
        grams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
        return len(set(grams)) / max(len(grams), 1)

    chars = list(blob)
    words = _WORD_RE.findall(blob)
    div = {
        "unique_char_bigram": unique_ratio(chars, 2),
        "unique_char_5gram": unique_ratio(chars, 5),
        "unique_word_unigram": len(set(words)) / max(len(words), 1),
        "unique_word_bigram": unique_ratio(words, 2),
        "vocab_words_in_samples": len(set(w.lower() for w in words)),
        "chars_generated": len(blob),
    }
    return fmt, div, samples


# ---------------------------------------------------------------------------
# Domain discrimination
# ---------------------------------------------------------------------------


def domain_texts() -> dict[str, str]:
    """Fixed probes: in-domain vs out-of-domain. No network required."""
    return {
        "shakespeare_like": (
            "ROMEO:\n"
            "But soft, what light through yonder window breaks?\n"
            "It is the east, and Juliet is the sun.\n"
            "Arise, fair sun, and kill the envious moon,\n"
            "Who is already sick and pale with grief.\n"
        ),
        "modern_prose": (
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning models require large amounts of data and compute. "
            "In this paper we propose a simple baseline that works surprisingly well.\n"
        ),
        "source_code": (
            "def train(model, data, steps):\n"
            "    for i in range(steps):\n"
            "        loss = model(data[i])\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "    return loss.item()\n"
        ),
        "random_ascii": "".join(
            chr(c) for c in np.random.default_rng(0).integers(32, 127, size=400)
        ),
        "repeated": ("to be or not to be " * 40),
    }


def domain_discrimination(model, stoi, block_size: int) -> dict:
    texts = domain_texts()
    scores = {name: nll_of_text(model, t, stoi, block_size) for name, t in texts.items()}
    # lower NLL on shakespeare_like than others is success
    base = scores["shakespeare_like"]
    gaps = {
        f"gap_vs_{name}": scores[name] - base
        for name in scores
        if name != "shakespeare_like"
    }
    # correct ranking: shakespeare should be best (lowest) among prose/code/random
    ordered = sorted(scores, key=scores.get)
    return {
        "nll": scores,
        "gaps": gaps,
        "shakespeare_rank": ordered.index("shakespeare_like") + 1,  # 1 = best
        "n_domains": len(scores),
        "prefers_shakespeare": ordered[0] == "shakespeare_like",
    }


# ---------------------------------------------------------------------------
# Famous-line cloze (optional cultural probe)
# ---------------------------------------------------------------------------


FAMOUS = [
    ("To be, or not to be, that is the ", "question"),
    ("ROMEO:\nBut soft, what light through yonder ", "window"),
    ("A horse! a horse! my kingdom for a ", "horse"),
    ("O Romeo, Romeo! wherefore art thou ", "Romeo"),
    ("All the world's a ", "stage"),
    ("Now is the winter of our ", "discontent"),
    ("Friends, Romans, countrymen, lend me your ", "ears"),
    ("What light through yonder window ", "breaks"),
]


@torch.no_grad()
def famous_cloze(model, stoi, itos, block_size: int, max_new: int = 24) -> dict:
    """
    Greedy-continue well-known prefixes; check if target word appears in
    the continuation (case-insensitive). Soft cultural probe, not a benchmark.
    """
    device = next(model.parameters()).device
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    hits = 0
    details = []
    for prefix, target in FAMOUS:
        # skip if prefix has OOV-heavy chars not in stoi — still try via get(,0)
        ids = torch.tensor(
            [[stoi.get(c, 0) for c in prefix]], dtype=torch.long, device=device
        )
        out = raw.generate(ids, max_new_tokens=max_new, temperature=0.2, top_k=10)
        cont = decode(out[0, ids.size(1) :], itos)
        ok = target.lower() in cont.lower() or target.lower() in (prefix + cont).lower()
        # also accept if next greedy chars match target start
        if not ok and cont.lower().startswith(target.lower()[:3]):
            ok = True
        hits += int(ok)
        details.append({"prefix": prefix[:40], "target": target, "cont": cont[:40], "hit": ok})
    return {
        "hit_rate": hits / max(len(FAMOUS), 1),
        "n": len(FAMOUS),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Cache claim
# ---------------------------------------------------------------------------


def cache_claim(cfg: GPTConfig, batch: int, optimizer: str, budget_scope: str) -> dict:
    """Product claim is C-inference WS vs L3; train tensor WS is informational."""
    from ll3m.model_torch import estimate_infer_ws

    hier = probe_cache()
    est = estimate_budget(cfg, batch, optimizer=optimizer)
    # Product pack is ternary Linears + int8 embeds (default of estimate_infer_ws).
    infer = estimate_infer_ws(cfg, quant="ternary", embed_quant="int8")
    scope = hier.l3_total_bytes if budget_scope == "chip" else hier.l3_bytes
    return {
        "l1d_kib": hier.l1d_bytes / 1024,
        "l2_kib": hier.l2_bytes / 1024,
        "l3_ccd_mib": hier.l3_bytes / (1024 * 1024),
        "l3_chip_mib": hier.l3_total_bytes / (1024 * 1024),
        "budget_scope": budget_scope,
        "scope_mib": scope / (1024 * 1024),
        # Primary product claim: bare-metal C inference
        "infer_ws_mib": infer.total / (1024 * 1024),
        "infer_weights_mib": infer.weight_bytes / (1024 * 1024),
        "infer_acts_mib": infer.activation_bytes / (1024 * 1024),
        "fits_scope": infer.total <= scope,
        "utilization": infer.total / scope,
        # Informational train accounting (not the L3 gate)
        "tensor_ws_mib": est.total / (1024 * 1024),
        "params_mib": est.bytes_params / (1024 * 1024),
        "n_params": est.n_params,
    }


# ---------------------------------------------------------------------------
# Capability index (single number, higher = better)
# ---------------------------------------------------------------------------


def capability_index(report: EvalReport) -> float:
    """
    Hand-weighted composite in [0, 1]. Transparent, not sacred.

    Components (weights sum to 1):
      0.35  compression: map BPC 0.8→1.0 ... 3.0→0.0  (char LM scale)
      0.15  top1 accuracy
      0.15  line completion char accuracy
      0.15  domain: prefers shakespeare * gap strength
      0.10  format: speaker presence * printable
      0.10  famous cloze hit rate
    """
    def clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    # BPC: excellent ~1.0-1.3, untrained ~4.1 (log2(65))
    bpc = report.val_bpc
    comp = clamp01((3.0 - bpc) / (3.0 - 0.9))

    top1 = clamp01(report.top1_acc / 0.70)  # 70% top1 is strong for char LM
    comp_acc = clamp01(report.completion.get("char_accuracy", 0.0) / 0.60)

    dom = report.domain
    gap = float(np.mean(list(dom.get("gaps", {}).values()))) if dom.get("gaps") else 0.0
    domain_s = clamp01((0.5 if dom.get("prefers_shakespeare") else 0.0) + gap / 4.0)

    fmt = report.format_score
    format_s = clamp01(
        0.5 * fmt.get("frac_samples_with_speaker", 0.0)
        + 0.3 * fmt.get("printable_ratio", 0.0)
        + 0.2 * clamp01(fmt.get("lines_per_sample", 0.0) / 15.0)
    )

    cloze = report.completion.get("famous_hit_rate", 0.0)
    # famous is nested: we store separately in completion or domain — see run_eval
    cloze = report.completion.get("famous_hit_rate", cloze)

    score = (
        0.35 * comp
        + 0.15 * top1
        + 0.15 * comp_acc
        + 0.15 * domain_s
        + 0.10 * format_s
        + 0.10 * clamp01(cloze)
    )
    return float(score)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> EvalReport:
    t0 = time.perf_counter()
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)

    model, stoi, itos, bpe = load_model(Path(args.checkpoint), device=str(device))
    # char-path helpers; BPE checkpoints still use stoi/itos maps from tokenizer
    if bpe is not None:
        stoi, itos = bpe.stoi, bpe.itos
    cfg = model.config
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = ckpt_meta.get("step")

    root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data) if args.data else root / "data" / "mix_val.txt"
    if not data_path.is_file():
        data_path = root / "data" / "tinyshakespeare.txt"
    text = load_text(data_path)
    # Tokenize with checkpoint BPE when available.
    if bpe is not None:
        data = torch.tensor(bpe.encode(text), dtype=torch.long)
    else:
        data = torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long)
    split = int(len(data) * 0.9) if len(data) > 10 else len(data) // 2
    train_t = data[:split].clone()
    val_t = data[split:].clone() if split < len(data) else data
    val_text = text  # qualitative probes use raw text + BPE encode

    print("=== ll3m capability eval ===")
    print(f"checkpoint: {args.checkpoint}  step={step}")
    print(
        f"model: L{cfg.n_layer} d{cfg.n_embd} h{cfg.n_head} T{cfg.block_size}  "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    print()

    # 1. compression
    print("[1/7] compression (val NLL / BPC)...")
    val_nll, val_tok = nll_over_data(model, val_t, cfg.block_size, batch_size=args.batch_size)
    train_nll, _ = nll_over_data(
        model, train_t, cfg.block_size, batch_size=args.batch_size, max_batches=40
    )
    val_bpc = val_nll / math.log(2)
    val_ppl = math.exp(min(val_nll, 20))
    print(f"  val NLL={val_nll:.4f}  BPC={val_bpc:.4f}  ppl={val_ppl:.2f}  ({val_tok:,} tok)")
    print(f"  train NLL={train_nll:.4f}  gap={val_nll - train_nll:+.4f}")

    # 2. domain
    print("[2/7] domain discrimination...")
    domain = domain_discrimination(model, stoi, cfg.block_size)
    for k, v in domain["nll"].items():
        print(f"  NLL[{k:16s}]={v:.4f}")
    print(
        f"  prefers_shakespeare={domain['prefers_shakespeare']}  "
        f"rank={domain['shakespeare_rank']}/{domain['n_domains']}"
    )

    # 3. top-k
    print("[3/7] next-char top-k accuracy...")
    topk = topk_accuracy(model, val_t, cfg.block_size, batch_size=args.batch_size)
    print(f"  top1={topk[1]*100:.2f}%  top5={topk[5]*100:.2f}%")

    # 4. line completion
    print("[4/7] line completion (val lines)...")
    comp = line_completion(
        model, val_text, stoi, itos, cfg.block_size, n_lines=args.n_lines
    )
    print(
        f"  exact={comp['exact_match']*100:.1f}%  "
        f"char_acc={comp['char_accuracy']*100:.1f}%  "
        f"suffix_bpc={comp['suffix_bpc']:.3f}"
    )

    # famous cloze
    print("  famous-line cloze...")
    cloze = famous_cloze(model, stoi, itos, cfg.block_size)
    comp["famous_hit_rate"] = cloze["hit_rate"]
    comp["famous_details"] = cloze["details"]
    print(f"  famous hit_rate={cloze['hit_rate']*100:.1f}%  ({cloze['n']} probes)")

    # 5–6. format + diversity
    print("[5/7] format + [6/7] diversity (free samples)...")
    prompts = [
        "ROMEO:\n",
        "JULIET:\n",
        "First Citizen:\n",
        "All:\n",
        "SCENE I. ",
    ]
    fmt, div, samples = format_and_diversity(
        model, stoi, itos, prompts, cfg.block_size, tokens_each=args.sample_tokens
    )
    print(
        f"  speaker_in_sample={fmt['frac_samples_with_speaker']*100:.0f}%  "
        f"printable={fmt['printable_ratio']*100:.1f}%  "
        f"lines/sample={fmt['lines_per_sample']:.1f}"
    )
    print(
        f"  unique 5-gram={div['unique_char_5gram']:.3f}  "
        f"word types={div['vocab_words_in_samples']}"
    )

    # 7. cache (C inference WS is the product claim)
    print("[7/7] cache claim (C infer WS)...")
    cache = cache_claim(cfg, batch=args.budget_batch, optimizer=args.optimizer, budget_scope=args.budget_scope)
    print(
        f"  C infer WS {cache['infer_ws_mib']:.2f} MiB / "
        f"{cache['scope_mib']:.1f} MiB {cache['budget_scope']} L3  "
        f"util={cache['utilization']*100:.1f}%  fits={cache['fits_scope']}"
    )
    print(
        f"  (train tensor WS {cache['tensor_ws_mib']:.2f} MiB is informational, not the gate)"
    )

    report = EvalReport(
        checkpoint=str(args.checkpoint),
        step=step,
        config=cfg.__dict__,
        n_params=sum(p.numel() for p in model.parameters()),
        val_nll=val_nll,
        val_bpc=val_bpc,
        val_ppl=val_ppl,
        train_nll=train_nll,
        domain=domain,
        top1_acc=topk[1],
        top5_acc=topk[5],
        completion=comp,
        format_score=fmt,
        diversity=div,
        cache=cache,
        samples=samples,
        eval_seconds=time.perf_counter() - t0,
        primary_score_bpc=val_bpc,
    )
    report.capability_index = capability_index(report)

    print()
    print("=== summary ===")
    print(f"  primary (val BPC):     {report.primary_score_bpc:.4f}   (lower better)")
    print(f"  capability index:      {report.capability_index:.3f}    (higher better, [0,1])")
    print(
        f"  cache fit (infer):     {cache['fits_scope']} "
        f"({cache['infer_ws_mib']:.1f}/{cache['scope_mib']:.1f} MiB)"
    )
    print(f"  eval wall time:        {report.eval_seconds:.1f}s")
    print()
    print("--- sample[0] ---")
    print(samples[0][:800] if samples else "(none)")
    print("-----------------")

    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate ll3m capabilities")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--budget-batch", type=int, default=8, help="batch used for WS accounting")
    p.add_argument(
        "--budget-scope",
        type=str,
        default="ccd",
        choices=["chip", "ccd"],
        help="L3 scope for C-infer cache claim (default: one CCD)",
    )
    p.add_argument("--optimizer", type=str, default="lion")
    p.add_argument("--n-lines", type=int, default=64)
    p.add_argument("--sample-tokens", type=int, default=400)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--json", type=str, default=None, help="write full report JSON")
    p.add_argument("--no-samples-in-json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    if not Path(args.checkpoint).is_file():
        print(f"missing checkpoint: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    report = run_eval(args)
    if args.json:
        d = asdict(report)
        if args.no_samples_in_json:
            d["samples"] = [f"<{len(report.samples)} samples omitted>"]
        # trim famous details if huge
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(d, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
