#!/usr/bin/env python3
"""Same MC log-likelihood protocol as run_std_benchmarks.py for HF causal LMs.

  uv run python scripts/run_std_benchmarks_hf.py \\
    --model Qwen/Qwen3.5-0.8B-Base --limit 400 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_ROOT / ".cache" / "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(_ROOT / ".cache" / "huggingface" / "datasets")
)

# Reuse task loaders
from run_std_benchmarks import load_task_examples  # type: ignore  # noqa: E402


@torch.no_grad()
def loglikelihood_hf(
    model,
    tokenizer,
    context: str,
    continuation: str,
    *,
    device: torch.device,
    max_len: int = 2048,
) -> float:
    cont = continuation if continuation.startswith((" ", "\n")) else " " + continuation
    # tokenize without special tokens for pure LM scoring
    ctx_ids = tokenizer.encode(context, add_special_tokens=False) if context else []
    cont_ids = tokenizer.encode(cont, add_special_tokens=False)
    if not cont_ids:
        return float("-inf")
    ids = ctx_ids + cont_ids
    if len(ids) < 2:
        return float("-inf")
    if len(ids) > max_len + 1:
        ids = ids[-(max_len + 1) :]
        n_cont = min(len(cont_ids), max_len)
    else:
        n_cont = len(cont_ids)

    t = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(t)
    logits = out.logits[0, :-1, :]
    logp = F.log_softmax(logits, dim=-1)
    targets = t[0, 1:]
    start = targets.numel() - n_cont
    if start < 0:
        start = 0
    lp = 0.0
    for i in range(start, targets.numel()):
        lp += logp[i, targets[i]].item()
    return lp


def pick_mc(model, tokenizer, ctx, choices, device, max_len) -> int:
    scores = [
        loglikelihood_hf(model, tokenizer, ctx, c, device=device, max_len=max_len)
        for c in choices
    ]
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def eval_task(model, tokenizer, examples, device, max_len) -> dict:
    correct = 0
    n = 0
    t0 = time.perf_counter()
    for ex in examples:
        choices = ex["choices"]
        if len(choices) < 2:
            continue
        pred = pick_mc(
            model, tokenizer, ex["context"], choices, device=device, max_len=max_len
        )
        if pred == int(ex["label"]):
            correct += 1
        n += 1
    elapsed = time.perf_counter() - t0
    random_base = sum(1.0 / len(ex["choices"]) for ex in examples) / max(n, 1)
    acc = correct / max(n, 1)
    return {
        "n": n,
        "correct": correct,
        "acc": acc,
        "acc_pct": 100.0 * acc,
        "random_pct": 100.0 * random_base,
        "elapsed_s": elapsed,
    }


def apply_weight_only_int4(model: torch.nn.Module) -> dict:
    """Per-row absmax int4 fake-quant of Linear weights (ROCm-friendly).

    Weights are dequantized back to the original dtype so matmul stays dense,
    but values are restricted to 16 levels (true 4-bit grid). Matches the
    common "weight-only 4-bit" quality regime when bnb NF4 is unavailable.
    """
    n_lin = 0
    n_elem = 0
    for mod in model.modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        w = mod.weight.data
        dtype = w.dtype
        wf = w.float()
        # per-output-channel scale over [-8, 7] (symmetric 4-bit)
        scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
        q = (wf / scale).round().clamp(-8, 7)
        mod.weight.data = (q * scale).to(dtype)
        n_lin += 1
        n_elem += w.numel()
    return {"linear_layers": n_lin, "elements": n_elem, "scheme": "per_row_absmax_int4"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,winogrande,boolq,openbookqa,sciq,copa",
    )
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="alias for --weight-int4 (portable per-row absmax int4; not NF4/NVFP4)",
    )
    ap.add_argument(
        "--weight-int4",
        action="store_true",
        help="per-row absmax int4 weight fake-quant (portable; product comparison default)",
    )
    args = ap.parse_args()

    # import path for sibling script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA unavailable → cpu")

    hip = getattr(torch.version, "hip", None)
    quant_meta: dict = {"mode": "none"}

    print(f"loading {args.model} on {device} hip={hip} …")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = None
    use_int4 = bool(args.weight_int4 or args.load_in_4bit)
    if use_int4:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model.to(device)
        quant_meta = apply_weight_only_int4(model)
        quant_meta["mode"] = "weight_only_int4_absmax"
        print(
            f"applied weight-only int4: {quant_meta['linear_layers']} Linear, "
            f"{quant_meta['elements']/1e9:.3f}B weights"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model.to(device)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}  ({n_params/1e9:.3f}B)  quant={quant_meta.get('mode')}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = {
        "model": args.model,
        "n_params": n_params,
        "quant": quant_meta,
        "device": str(device),
        "hip": hip,
        "limit": args.limit,
        "seed": args.seed,
        "protocol": "zero-shot continuation log-likelihood MC (same as ll3m harness)",
        "tasks": {},
    }
    t_all = time.perf_counter()
    for task in tasks:
        print(f"\n=== {task} (limit={args.limit}) ===")
        try:
            examples = load_task_examples(task, args.limit, args.seed)
        except Exception as e:
            print(f"  SKIP {e}")
            results["tasks"][task] = {"error": str(e)}
            continue
        print(f"  loaded {len(examples)}")
        met = eval_task(model, tok, examples, device, args.max_len)
        results["tasks"][task] = met
        print(
            f"  acc={met['acc_pct']:.1f}% (random~{met['random_pct']:.1f}%) "
            f"n={met['n']} {met['elapsed_s']:.1f}s"
        )

    accs = [
        v["acc_pct"]
        for v in results["tasks"].values()
        if isinstance(v, dict) and "acc_pct" in v
    ]
    results["macro_acc_pct"] = sum(accs) / max(len(accs), 1)
    results["elapsed_s"] = time.perf_counter() - t_all
    qtag = quant_meta.get("mode", "none")
    out = Path(
        args.out
        or (
            _ROOT
            / "checkpoints"
            / "compare"
            / f"{args.model.replace('/', '__')}_{qtag}_limit{args.limit}.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nmacro {results['macro_acc_pct']:.1f}%  wrote {out}")


if __name__ == "__main__":
    main()
