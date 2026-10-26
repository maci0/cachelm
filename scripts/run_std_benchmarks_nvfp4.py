#!/usr/bin/env python3
"""Std MC log-likelihood protocol for modelopt NVFP4 HF checkpoints.

Primary targets (GB10-friendly NVFP4 packs):
  osoleve/Qwen3.5-9B-Base-Text-NVFP4   # Base, fairer closed-book
  AxionML/Qwen3.5-9B-NVFP4             # post-trained 9B NVFP4 (popular)

  .venv/bin/python scripts/run_std_benchmarks_nvfp4.py \\
    --model osoleve/Qwen3.5-9B-Base-Text-NVFP4 --limit 400
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
os.environ.setdefault("TRITON_CACHE_DIR", str(_ROOT / ".cache" / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_ROOT / ".cache" / "torchinductor"))

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_std_benchmarks import load_task_examples  # noqa: E402


@torch.no_grad()
def loglikelihood_hf(model, tokenizer, context: str, continuation: str, *, device, max_len: int) -> float:
    cont = continuation if continuation.startswith((" ", "\n")) else " " + continuation
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
    logp = F.log_softmax(logits.float(), dim=-1)
    targets = t[0, 1:]
    start = max(0, targets.numel() - n_cont)
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
        pred = pick_mc(model, tokenizer, ex["context"], choices, device, max_len)
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


def load_nvfp4_model(model_id: str, device: str):
    """Load modelopt NVFP4 checkpoint; register modelopt with transformers if needed."""
    # Import modelopt so quant_method=modelopt is registered when available.
    try:
        import modelopt.torch.quantization  # noqa: F401
        import modelopt.torch.opt  # noqa: F401

        print("modelopt imported", flush=True)
    except Exception as e:
        print(f"modelopt import note: {e}", flush=True)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    qc = getattr(cfg, "quantization_config", None)
    print(f"quantization_config type={type(qc).__name__}  qc={qc}", flush=True)

    # Prefer auto device map for multi-GB packs
    kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e1:
        print(f"AutoModelForCausalLM failed ({e1}); trying ConditionalGeneration…", flush=True)
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    model.eval()
    # Resolve primary device for tensor placement
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
    return model, tok, dev, qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=str,
        default="osoleve/Qwen3.5-9B-Base-Text-NVFP4",
        help="HF id of NVFP4 checkpoint (modelopt)",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,winogrande,boolq,openbookqa,sciq,copa",
    )
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print(f"loading NVFP4 model {args.model}", flush=True)
    model, tok, device, qc = load_nvfp4_model(args.model, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} ({n_params/1e9:.3f}B)", flush=True)

    # smoke
    with torch.no_grad():
        ids = tok("Hello", return_tensors="pt")
        ids = {k: v.to(device) for k, v in ids.items()}
        # some multimodal models want input_ids only
        out = model(input_ids=ids["input_ids"])
        logits = out.logits if hasattr(out, "logits") else out[0]
        print(f"smoke logits {tuple(logits.shape)}", flush=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = {
        "model": args.model,
        "n_params": n_params,
        "quant": {
            "mode": "nvfp4_modelopt",
            "quantization_config": (
                qc if isinstance(qc, dict) else getattr(qc, "to_dict", lambda: str(qc))()
            ),
        },
        "device": str(device),
        "limit": args.limit,
        "seed": args.seed,
        "protocol": "zero-shot continuation log-likelihood MC (same as ll3m harness)",
        "tasks": {},
    }

    # wrap model call for logits-only API used by loglikelihood
    class _LogitsModel:
        def __init__(self, m):
            self.m = m

        def __call__(self, t):
            o = self.m(input_ids=t)
            if hasattr(o, "logits"):
                return o
            # tuple fallback
            class R:
                pass

            r = R()
            r.logits = o[0] if isinstance(o, (tuple, list)) else o
            return r

    mwrap = _LogitsModel(model)

    t_all = time.perf_counter()
    for task in tasks:
        print(f"\n=== {task} (limit={args.limit}) ===", flush=True)
        try:
            examples = load_task_examples(task, args.limit, args.seed)
        except Exception as e:
            print(f"  SKIP {e}", flush=True)
            results["tasks"][task] = {"error": str(e)}
            continue
        print(f"  loaded {len(examples)}", flush=True)
        met = eval_task(mwrap, tok, examples, device, args.max_len)
        results["tasks"][task] = met
        print(
            f"  acc={met['acc_pct']:.1f}% (random~{met['random_pct']:.1f}%) "
            f"n={met['n']} {met['elapsed_s']:.1f}s",
            flush=True,
        )

    accs = [
        v["acc_pct"]
        for v in results["tasks"].values()
        if isinstance(v, dict) and "acc_pct" in v
    ]
    results["macro_acc_pct"] = sum(accs) / max(len(accs), 1)
    results["elapsed_s"] = time.perf_counter() - t_all

    # Compare to LL3M denser continue if present
    ll3m_path = _ROOT / "checkpoints/v76_dense_c/std_benchmarks_expC_limit400.json"
    if ll3m_path.is_file():
        ll3m = json.loads(ll3m_path.read_text())
        results["ll3m_C_macro"] = ll3m.get("macro_acc_pct")
        results["delta_ll3mC_minus_this"] = (
            (ll3m.get("macro_acc_pct") or 0) - results["macro_acc_pct"]
        )

    safe = args.model.replace("/", "__")
    out = Path(
        args.out
        or (_ROOT / "checkpoints" / "compare" / f"{safe}_nvfp4_limit{args.limit}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # make quant json-serializable
    try:
        json.dumps(results["quant"])
    except TypeError:
        results["quant"] = {"mode": "nvfp4_modelopt", "raw": str(qc)}
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nmacro {results['macro_acc_pct']:.2f}%  wrote {out}", flush=True)
    if "ll3m_C_macro" in results:
        print(
            f"LL3M C macro={results['ll3m_C_macro']:.2f}%  "
            f"Δ(ll3m-qwen)={results['delta_ll3mC_minus_this']:+.2f} pp",
            flush=True,
        )


if __name__ == "__main__":
    main()
