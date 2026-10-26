#!/usr/bin/env python3
"""Industry-standard zero-shot multiple-choice benchmarks for ll3m.

Scores continuation log-likelihood (Eleuther-style): for each option, compute
mean NLL of the choice tokens given the context; pick lowest NLL.

Tasks (HF datasets):
  hellaswag, arc_easy, winogrande, boolq, openbookqa, sciq, copa
  (PIQA omitted: HF script datasets blocked on datasets>=4)

Note: this is a ~79M ternary V=2048 domain model. Expect near-random on many
tasks; the point is a standard comparable number, not frontier scores.

  export HF_HOME=$PWD/.cache/huggingface
  uv run python scripts/run_std_benchmarks.py \\
    --checkpoint checkpoints/v76_base_en_tool/best.pt \\
    --tokenizer data/base_en/tokenizer.json \\
    --limit 300 --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Prefer project-local HF cache
_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_ROOT / ".cache" / "huggingface" / "datasets"))

from ll3m.tokenizer import BPETokenizer  # noqa: E402
from ll3m.train import load_model  # noqa: E402


@torch.no_grad()
def loglikelihood(
    model,
    bpe: BPETokenizer,
    context: str,
    continuation: str,
    *,
    block_size: int,
    device: torch.device,
) -> float:
    """Sum log p(continuation tokens | context). Higher is better."""
    if not continuation:
        return float("-inf")
    # Leading space often matches training (Eleuther convention)
    cont = continuation if continuation.startswith((" ", "\n")) else " " + continuation
    ctx_ids = bpe.encode(context) if context else []
    cont_ids = bpe.encode(cont)
    if not cont_ids:
        return float("-inf")
    ids = ctx_ids + cont_ids
    # Need at least 2 tokens for a prediction
    if len(ids) < 2:
        return float("-inf")
    # Crop to block: keep last block_size+1 tokens of full sequence
    if len(ids) > block_size + 1:
        ids = ids[-(block_size + 1) :]
        # how many cont tokens remain after crop
        n_cont = min(len(cont_ids), block_size)
    else:
        n_cont = len(cont_ids)

    t = torch.tensor(ids, dtype=torch.long, device=device)
    x = t[:-1].unsqueeze(0)
    # logits for positions predicting each next token
    logits, _ = model(x)
    # log probs
    logp = F.log_softmax(logits[0], dim=-1)
    # targets are t[1:]; we want only the continuation targets at the end
    targets = t[1:]
    # continuation occupies the last n_cont predictions
    start = targets.numel() - n_cont
    if start < 0:
        start = 0
        n_cont = targets.numel()
    lp = 0.0
    for i in range(start, targets.numel()):
        lp += logp[i, targets[i]].item()
    return lp


def pick_mc(model, bpe, ctx: str, choices: list[str], block_size: int, device) -> int:
    scores = [
        loglikelihood(model, bpe, ctx, c, block_size=block_size, device=device)
        for c in choices
    ]
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def load_task_examples(task: str, limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    examples: list[dict] = []

    if task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:limit]:
            row = ds[i]
            ctx = row["ctx"]
            endings = list(row["endings"])
            label = int(row["label"])
            examples.append(
                {"context": ctx, "choices": endings, "label": label, "task": task}
            )

    elif task == "sciq":
        ds = load_dataset("allenai/sciq", split="validation")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:limit]:
            row = ds[i]
            # 4-way: correct + 3 distractors (shuffle options)
            choices = [
                row["correct_answer"],
                row["distractor1"],
                row["distractor2"],
                row["distractor3"],
            ]
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [choices[j] for j in order]
            label = order.index(0)
            support = (row.get("support") or "")[:200]
            ctx = f"{support}\nQuestion: {row['question']}\nAnswer:" if support else f"Question: {row['question']}\nAnswer:"
            examples.append(
                {
                    "context": ctx,
                    "choices": shuffled,
                    "label": label,
                    "task": task,
                }
            )

    elif task == "copa":
        ds = load_dataset("aps/super_glue", "copa", split="validation")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:limit]:
            row = ds[i]
            q = row["question"]  # cause | effect
            prem = row["premise"]
            choices = [row["choice1"], row["choice2"]]
            label = int(row["label"])
            ctx = f"Premise: {prem}\nWhat was the {q}?\nAnswer:"
            examples.append(
                {
                    "context": ctx,
                    "choices": choices,
                    "label": label,
                    "task": task,
                }
            )

    elif task == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:limit]:
            row = ds[i]
            stem = row["question"]
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            # map answerKey to index
            key = str(row["answerKey"])
            if key not in labels:
                continue
            label = labels.index(key)
            ctx = f"Question: {stem}\nAnswer:"
            examples.append(
                {"context": ctx, "choices": list(texts), "label": label, "task": task}
            )

    elif task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:limit]:
            row = ds[i]
            sent = row["sentence"]
            # sentence has _
            o1, o2 = row["option1"], row["option2"]
            c1 = sent.replace("_", o1)
            c2 = sent.replace("_", o2)
            # score full sentence as continuation of empty / short prefix
            label = int(row["answer"]) - 1  # 1/2 → 0/1
            examples.append(
                {
                    "context": "",
                    "choices": [c1, c2],
                    "label": label,
                    "task": task,
                }
            )

    elif task == "boolq":
        ds = load_dataset("google/boolq", split="validation")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:limit]:
            row = ds[i]
            passage = row["passage"]
            question = row["question"]
            # truncate passage for T=256
            if len(passage) > 400:
                passage = passage[:400]
            ctx = f"{passage}\nQuestion: {question}?\nAnswer:"
            label = 1 if row["answer"] else 0
            examples.append(
                {
                    "context": ctx,
                    "choices": [" no", " yes"],
                    "label": label,
                    "task": task,
                }
            )

    elif task == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", split="test")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:limit]:
            row = ds[i]
            stem = row["question_stem"]
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            key = str(row["answerKey"])
            if key not in labels:
                continue
            label = labels.index(key)
            examples.append(
                {
                    "context": f"Question: {stem}\nAnswer:",
                    "choices": list(texts),
                    "label": label,
                    "task": task,
                }
            )
    else:
        raise ValueError(f"unknown task {task}")

    return examples


def eval_task(model, bpe, examples: list[dict], block_size: int, device) -> dict:
    correct = 0
    n = 0
    n_choices_sum = 0
    t0 = time.perf_counter()
    for ex in examples:
        choices = ex["choices"]
        if len(choices) < 2:
            continue
        pred = pick_mc(
            model, bpe, ex["context"], choices, block_size=block_size, device=device
        )
        if pred == int(ex["label"]):
            correct += 1
        n += 1
        n_choices_sum += len(choices)
    elapsed = time.perf_counter() - t0
    acc = correct / max(n, 1)
    random_base = n / max(n_choices_sum, 1)  # average 1/k
    # better random: mean of 1/|choices|
    random_base = sum(1.0 / len(ex["choices"]) for ex in examples) / max(n, 1)
    return {
        "n": n,
        "correct": correct,
        "acc": acc,
        "acc_pct": 100.0 * acc,
        "random_pct": 100.0 * random_base,
        "elapsed_s": elapsed,
    }


def resolve_bpe(ckpt_path: Path, tokenizer: str | None) -> BPETokenizer:
    if tokenizer:
        return BPETokenizer.load(tokenizer)
    # try base_en then story
    for p in (
        _ROOT / "data" / "base_en" / "tokenizer.json",
        _ROOT / "data" / "tokenizer.json",
    ):
        if p.is_file():
            return BPETokenizer.load(p)
    raise SystemExit("need --tokenizer")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v76_base_en_tool/best.pt",
    )
    ap.add_argument("--tokenizer", type=str, default="data/base_en/tokenizer.json")
    ap.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,winogrande,boolq,openbookqa,sciq,copa",
    )
    ap.add_argument("--limit", type=int, default=300, help="examples per task")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"missing {ck}")

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using cpu")
        device = torch.device("cpu")

    print(f"loading {ck} on {device} …")
    model, _stoi, _itos, bpe = load_model(ck, device=str(device))
    if bpe is None:
        bpe = resolve_bpe(ck, args.tokenizer)
    else:
        # prefer explicit tokenizer if paths differ / remote path missing
        tok_p = Path(args.tokenizer) if args.tokenizer else None
        if tok_p and tok_p.is_file():
            bpe = BPETokenizer.load(tok_p)
    model.eval()
    block = int(model.config.block_size)
    print(f"model L{model.config.n_layer} d{model.config.n_embd} T{block} V={bpe.vocab_size}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results: dict = {
        "checkpoint": str(ck),
        "tokenizer": args.tokenizer,
        "device": str(device),
        "limit": args.limit,
        "seed": args.seed,
        "tasks": {},
        "notes": (
            "Zero-shot continuation log-likelihood MC. "
            "~79M ternary V=2048 domain LM; near-random is expected on many tasks."
        ),
    }

    t_all = time.perf_counter()
    for task in tasks:
        print(f"\n=== {task} (limit={args.limit}) ===")
        try:
            examples = load_task_examples(task, args.limit, args.seed)
        except Exception as e:
            print(f"  SKIP load error: {e}")
            results["tasks"][task] = {"error": str(e)}
            continue
        print(f"  loaded {len(examples)} examples")
        met = eval_task(model, bpe, examples, block, device)
        results["tasks"][task] = met
        print(
            f"  acc={met['acc_pct']:.1f}%  "
            f"(random~{met['random_pct']:.1f}%)  "
            f"n={met['n']}  {met['elapsed_s']:.1f}s"
        )

    results["elapsed_s"] = time.perf_counter() - t_all
    # macro average over successful tasks
    accs = [
        v["acc_pct"]
        for v in results["tasks"].values()
        if isinstance(v, dict) and "acc_pct" in v
    ]
    results["macro_acc_pct"] = sum(accs) / max(len(accs), 1)

    out = Path(
        args.out
        or (ck.parent / f"std_benchmarks_limit{args.limit}.json")
    )
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n=== macro acc {results['macro_acc_pct']:.1f}% over {len(accs)} tasks ===")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
