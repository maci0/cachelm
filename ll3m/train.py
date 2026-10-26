"""Train a GPT for max capability under *inference* L3 (C int8 + KV).

Training may use full DRAM. The product constraint is bare-metal C infer WS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ll3m.cache_info import budget_bytes, probe_cache
from ll3m.config import GPTConfig
from ll3m.model_torch import (
    GPT,
    Lion,
    choose_config_infer,
    estimate_infer_ws,
    working_set_bytes,
)
from ll3m.tokenizer import BPETokenizer


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def decode(ids, itos: dict[int, str]) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.view(-1).tolist()
    return "".join(itos[int(i)] for i in ids)


def get_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def cosine_lr(step: int, total: int, base: float, warmup: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * t)))


def resolve_device(spec: str) -> torch.device:
    """cpu | cuda | cuda:N | hip | rocm | auto. ROCm uses the cuda API surface."""
    s = (spec or "auto").lower()
    if s in ("auto",):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if s in ("rocm", "hip"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "ROCm/HIP requested but torch.cuda.is_available() is False. "
                "Install a ROCm build of PyTorch "
                "(e.g. uv pip install torch==2.10.0+rocm7.0 "
                "--index-url https://download.pytorch.org/whl/rocm7.0)."
            )
        return torch.device("cuda")
    if s.startswith("cuda") or s == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit(f"device {spec!r} requested but CUDA/ROCm is not available")
        return torch.device("cuda" if s == "gpu" else s)
    if s == "cpu":
        return torch.device("cpu")
    return torch.device(spec)


def configure_gpu_memory(device: torch.device, fraction: float) -> None:
    """
    Cap process VRAM so the desktop / other apps still have headroom.

    fraction in (0, 1]: hard per-process limit via set_per_process_memory_fraction.
    Also sets allocator env defaults that reduce fragmentation on ROCm/CUDA.
    """
    if device.type != "cuda":
        return
    frac = float(fraction)
    if not (0.05 <= frac <= 1.0):
        raise SystemExit(f"--gpu-mem-fraction must be in [0.05, 1.0], got {frac}")
    # Prefer expandable segments; cap split size to avoid huge reserved slabs.
    os.environ.setdefault(
        "PYTORCH_HIP_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
    )
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
    )
    idx = device.index if device.index is not None else torch.cuda.current_device()
    try:
        torch.cuda.set_per_process_memory_fraction(frac, device=idx)
    except Exception as e:
        print(f"warning: set_per_process_memory_fraction failed: {e}", file=sys.stderr)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    props = torch.cuda.get_device_properties(idx)
    total_gib = props.total_memory / (1024**3)
    print(
        f"GPU memory cap: {frac:.0%} of {total_gib:.1f} GiB "
        f"(~{frac * total_gib:.1f} GiB max for this process)  device={idx}"
    )


def train(args: argparse.Namespace) -> None:
    device = resolve_device(getattr(args, "device", "auto"))
    if device.type == "cuda":
        configure_gpu_memory(device, getattr(args, "gpu_mem_fraction", 0.45))
    # CPU thread pool only matters on CPU; keep modest on GPU to leave cores free.
    default_threads = 4 if device.type == "cuda" else (
        16 if getattr(args, "budget_scope", "chip") == "chip" else 8
    )
    nthreads = args.threads or min(default_threads, os.cpu_count() or 8)
    torch.set_num_threads(nthreads)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", str(nthreads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(nthreads))

    root = Path(__file__).resolve().parent.parent
    # Prefer prepared mix corpus; fall back to Tiny Shakespeare chars.
    default_data = root / "data" / "mix_train.txt"
    if not default_data.is_file():
        default_data = root / "data" / "tinyshakespeare.txt"
    data_path = Path(args.data) if args.data else default_data
    if not data_path.is_file():
        print(f"missing data file: {data_path}", file=sys.stderr)
        sys.exit(1)
    val_path = Path(args.val_data) if args.val_data else None
    if val_path is None:
        cand = data_path.parent / "mix_val.txt"
        if cand.is_file() and data_path.name.startswith("mix"):
            val_path = cand

    hier = probe_cache()
    # Inference L3 is the product gate (default: one CCD).
    infer_scope = hier.l3_total_bytes if args.infer_scope == "chip" else hier.l3_bytes
    infer_scope_name = (
        "chip L3 total" if args.infer_scope == "chip" else "one L3 instance (CCD)"
    )
    infer_budget = args.infer_budget_bytes or int(infer_scope * args.infer_budget_fraction)
    # Train budget is informational unless --enforce-train-budget.
    train_budget = args.budget_bytes or budget_bytes(
        hier, fraction=args.budget_fraction, scope=args.budget_scope
    )
    train_scope = hier.l3_total_bytes if args.budget_scope == "chip" else hier.l3_bytes
    train_scope_name = (
        "chip L3 total" if args.budget_scope == "chip" else "one L3 instance (CCD)"
    )

    print("=== CPU cache (infer product gate) ===")
    print(hier.summary())
    print(
        f"INFER L3 gate: {infer_budget / (1024 * 1024):.2f} MiB "
        f"({args.infer_budget_fraction:.0%} of {infer_scope_name} = "
        f"{infer_scope / (1024 * 1024):.1f} MiB)  ← product constraint"
    )
    print(
        f"train tensor estimate budget (info): {train_budget / (1024 * 1024):.2f} MiB "
        f"({args.budget_fraction:.0%} of {train_scope_name}); "
        f"enforced={args.enforce_train_budget}"
    )
    hip = getattr(torch.version, "hip", None)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(
        f"backend: torch {torch.__version__}  device={device} ({dev_name})  "
        f"hip={hip}  threads={nthreads}  optimizer={args.optimizer}"
    )
    print()
    tok_path = Path(args.tokenizer) if args.tokenizer else root / "data" / "tokenizer.json"
    # Accept either .txt (prefer sibling .bin if present) or a pre-encoded .bin.
    if data_path.suffix == ".bin":
        train_bin: Path | None = data_path if data_path.is_file() else None
    else:
        cand = data_path.with_suffix(".bin")
        train_bin = cand if cand.is_file() else None
    if not tok_path.is_file():
        print(
            f"missing tokenizer {tok_path}; run scripts/prepare_data.py first",
            file=sys.stderr,
        )
        sys.exit(1)
    bpe = BPETokenizer.load(tok_path)
    stoi, itos = bpe.stoi, bpe.itos
    vocab_size = bpe.vocab_size
    if train_bin is not None and train_bin.is_file():
        print(f"loading pre-encoded tokens {train_bin.name}...")
        data = np.fromfile(train_bin, dtype=np.uint16).astype(np.int64)
    else:
        text = load_text(data_path)
        print(f"encoding corpus with BPE ({bpe.vocab_size} tokens)...")
        data = np.array(bpe.encode(text), dtype=np.int64)

    if val_path is not None and val_path.is_file():
        val_bin = val_path.with_suffix(".bin")
        if val_bin.is_file():
            vdata = np.fromfile(val_bin, dtype=np.uint16).astype(np.int64)
        else:
            vdata = np.array(bpe.encode(load_text(val_path)), dtype=np.int64)
        # Keep corpora on CPU (or pinned host RAM). Shipping the whole stream to
        # VRAM is a common cause of GPU thrash / system freezes on long runs.
        train_data = torch.from_numpy(np.ascontiguousarray(data))
        val_data = torch.from_numpy(np.ascontiguousarray(vdata))
        print(
            f"corpus: train={data_path.name} ({len(data):,} tok)  "
            f"val={val_path.name} ({len(vdata):,} tok)  vocab={vocab_size}"
        )
    else:
        n = len(data)
        split = int(n * 0.9)
        train_data = torch.from_numpy(np.ascontiguousarray(data[:split]))
        val_data = torch.from_numpy(np.ascontiguousarray(data[split:]))
        print(f"corpus: {data_path.name}  {n:,} tok  vocab={vocab_size}  (90/10 split)")

    if device.type == "cuda":
        try:
            train_data = train_data.pin_memory()
            val_data = val_data.pin_memory()
            print("corpus: host pinned memory (batches only move to GPU)")
        except RuntimeError:
            print("corpus: host RAM (pin_memory unavailable)")

    if args.resume:
        ck_preview = torch.load(args.resume, map_location="cpu", weights_only=False)
        c = dict(ck_preview["config"])
        c["dropout"] = args.dropout
        cfg = GPTConfig.from_dict(c)
        if c["vocab_size"] != vocab_size:
            print(
                f"warning: checkpoint vocab {c['vocab_size']} != data vocab {vocab_size}",
                file=sys.stderr,
            )
    elif args.n_embd:
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            n_kv_head=args.n_kv_head,
            intermediate_size=args.intermediate_size,
            rope_theta=args.rope_theta,
            rope_partial=args.rope_partial,
            logit_softcap=args.logit_softcap,
            peri_ln=args.peri_ln,
            layer_scale_init=args.layer_scale_init,
        )
    else:
        cfg, _ = choose_config_infer(
            vocab_size,
            infer_budget,
            block_size=args.block_size,
        )
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=args.dropout,
            n_kv_head=cfg.n_kv_head,
            intermediate_size=cfg.intermediate_size,
            rope_theta=cfg.rope_theta,
            rope_partial=args.rope_partial,
            logit_softcap=args.logit_softcap,
            peri_ln=args.peri_ln,
            layer_scale_init=args.layer_scale_init,
        )

    from ll3m.model_torch import estimate_budget

    detail = estimate_budget(cfg, args.batch_size, optimizer=args.optimizer)
    # Product serve quant (default ternary) drives the L3 gate; int8 printed for compare.
    serve_q = getattr(args, "serve_quant", "ternary")
    infer = estimate_infer_ws(cfg, quant=serve_q)
    infer_i8 = estimate_infer_ws(cfg, quant="int8")
    print("=== model (max capability under INFER L3) ===")
    print(
        f"arch={cfg.arch}  layers={cfg.n_layer}  embd={cfg.n_embd}  "
        f"heads={cfg.n_head}/{cfg.kv_head}  mid={cfg.intermediate}  "
        f"block={cfg.block_size}  vocab={cfg.vocab_size}  "
        f"rope_partial={cfg.rope_partial}  softcap={cfg.logit_softcap}  "
        f"peri_ln={cfg.peri_ln}  layer_scale={cfg.layer_scale_init}  "
        f"z_loss={args.z_loss}  mtp={args.mtp}/{args.mtp_depth}"
    )
    print(f"parameters (formula): {cfg.n_params():,}")
    print(
        f"C infer WS ({serve_q}+KV): {infer.total / (1024 * 1024):.2f} MiB  "
        f"(weights {infer.weight_bytes / (1024 * 1024):.2f} + "
        f"acts {infer.activation_bytes / (1024 * 1024):.2f})  "
        f"[int8 est {infer_i8.total / (1024 * 1024):.2f} MiB]"
    )
    fits_infer = infer.total <= infer_budget
    print(
        f"fits INFER L3: {'YES' if fits_infer else 'NO'}  "
        f"({infer.total / infer_budget * 100:.1f}% of "
        f"{infer_budget / (1024 * 1024):.2f} MiB)  gate={serve_q}"
    )
    print(
        f"train tensor WS (info): {detail.total / (1024 * 1024):.2f} MiB  "
        f"(may exceed L3; DRAM OK)"
    )
    if not fits_infer and not args.force:
        print(
            "refusing: C infer WS exceeds infer L3 gate (pass --force to override)",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.enforce_train_budget and detail.total > train_budget and not args.force:
        print(
            "refusing: train WS over train budget with --enforce-train-budget "
            "(pass --force to override)",
            file=sys.stderr,
        )
        sys.exit(2)
    print()
    torch.manual_seed(args.seed)
    start_step = 0
    resumed_best_val = float("inf")
    resumed_qat = "none"
    resumed_embed_qat = "none"
    model = GPT(cfg).to(device)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            print(f"resume checkpoint missing: {resume_path}", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["params"])
        start_step = int(ckpt.get("step", 0))
        resumed_qat = str(ckpt.get("qat", "none") or "none")
        resumed_embed_qat = str(ckpt.get("embed_qat", "none") or "none")
        if "val_loss" in ckpt:
            resumed_best_val = float(ckpt["val_loss"])
        print(f"resumed weights from {resume_path} (step {start_step})", flush=True)

    if args.grad_checkpoint:
        _enable_checkpointing(model)

    qat_mode = getattr(args, "qat", "none") or "none"
    embed_qat_mode = getattr(args, "embed_qat", "none") or "none"
    if qat_mode != "none":
        from ll3m.qat import enable_qat_on_model

        n_q = enable_qat_on_model(
            model,
            num_bits=8 if qat_mode == "int8" else 2,
            ternary=(qat_mode == "ternary"),
            binary=(qat_mode == "binary"),
        )
        print(f"QAT enabled: mode={qat_mode}  patched Linear modules={n_q}")
    if embed_qat_mode != "none":
        from ll3m.qat import enable_embed_qat_on_model

        enable_embed_qat_on_model(model, mode=embed_qat_mode)
        print(f"embed QAT enabled: mode={embed_qat_mode}", flush=True)
    # Val NLL under fake-quant is not comparable to a different QAT/fp32 bar.
    prev_best = resumed_best_val
    resumed_best_val = resolve_resumed_best_val(
        resumed_best_val,
        resumed_qat,
        qat_mode,
        resumed_embed_qat=resumed_embed_qat,
        embed_qat_mode=embed_qat_mode,
    )
    if resumed_best_val != prev_best and math.isinf(resumed_best_val):
        print(
            f"QAT mode change (lin {resumed_qat}→{qat_mode}, "
            f"embed {resumed_embed_qat}→{embed_qat_mode}): "
            f"reset best_val (was {prev_best:.4f}) for early-stop / best.pt",
            flush=True,
        )
    # Persist QAT tags on the module so save_checkpoint records them
    raw_for_tag = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_for_tag._ll3m_qat = qat_mode  # type: ignore[attr-defined]
    raw_for_tag._ll3m_embed_qat = embed_qat_mode  # type: ignore[attr-defined]

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if args.optimizer == "lion":
        opt = Lion(param_groups, lr=args.lr, betas=(0.9, 0.99))
    else:
        opt = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
        )

    ws = working_set_bytes(model, args.optimizer, args.batch_size)
    print("=== working sets ===")
    print(
        f"  C infer WS        {infer.total / (1024 * 1024):7.3f} MiB  "
        f"({'FITS' if infer.total <= infer_budget else 'OVER'} in "
        f"{infer_budget / (1024 * 1024):.1f} MiB infer L3)"
    )
    for k, v in ws.items():
        if k == "n_params":
            print(f"  train {k:11s} {v:,}")
        else:
            print(f"  train {k:11s} {v / (1024 * 1024):7.3f} MiB")
    print(f"  process VmRSS     {get_rss_bytes() / (1024 * 1024):7.3f} MiB")
    print()

    def get_batch(split_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample windows on host; copy only the microbatch to GPU."""
        T = cfg.block_size
        B = args.batch_size
        ix = torch.randint(0, len(split_t) - T - 1, (B,))
        # Vectorized gather beats Python stack of slices
        offs = torch.arange(T)
        x = split_t[ix.unsqueeze(1) + offs]
        y = split_t[ix.unsqueeze(1) + offs + 1]
        if device.type == "cuda":
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    @torch.no_grad()
    def eval_loss(split_t: torch.Tensor, iters: int = 50) -> float:
        model.eval()
        losses = []
        for _ in range(iters):
            x, y = get_batch(split_t)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float(sum(losses) / len(losses))

    out_dir = Path(args.out_dir) if args.out_dir else root / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_step = args.batch_size * cfg.block_size * args.grad_accum
    print(
        f"training {args.steps} steps  microbatch={args.batch_size}  "
        f"accum={args.grad_accum}  ({tokens_per_step} tok/step)  "
        f"lr={args.lr}  warmup={args.warmup}"
    )
    print(f"~{args.steps * tokens_per_step / 1e6:.1f}M tokens total")

    # Speed probe only: measure fwd+bwd throughput. Never call opt.step() here
    # (that used to corrupt init weights + Lion moments before real training).
    model.train()
    x, y = get_batch(train_data)
    for _ in range(3):
        _, loss = model(x, y)
        loss.backward()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _, loss = model(x, y)
        loss.backward()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    probe_tps = 10 * args.batch_size * cfg.block_size / (time.perf_counter() - t0)
    print(f"speed probe: {probe_tps:.0f} tok/s  (no optimizer steps)")
    print()

    t0 = time.perf_counter()
    log: list[dict] = []
    best_val = resumed_best_val
    bad_evals = 0
    running = 0.0
    opt.zero_grad(set_to_none=True)

    # continue past resume step; cosine schedule uses absolute step index
    first_step = start_step + 1
    if first_step > args.steps:
        print(f"already at step {start_step} >= {args.steps}; nothing to do")
        return

    for step in range(first_step, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr

        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data)
            _, loss = model(
                x,
                y,
                z_loss_coeff=args.z_loss,
                mtp_weight=args.mtp,
                mtp_depth=args.mtp_depth,
            )
            (loss / args.grad_accum).backward()
            loss_acc += loss.item()
        loss_val = loss_acc / args.grad_accum
        running = loss_val if step == first_step else (0.95 * running + 0.05 * loss_val)

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == first_step:
            elapsed = time.perf_counter() - t0
            steps_done = step - start_step
            tok_s = steps_done * tokens_per_step / max(elapsed, 1e-9)
            msg = (
                f"step {step:5d}/{args.steps}  loss {loss_val:.4f}  "
                f"avg {running:.4f}  |g| {float(gn):.2f}  lr {lr:.2e}  {tok_s:.0f} tok/s"
            )
            print(msg, flush=True)
            entry = {
                "step": step,
                "train_loss": loss_val,
                "avg_loss": running,
                "grad_norm": float(gn),
                "lr": lr,
                "tok_s": tok_s,
            }
            log.append(entry)
            (out_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "steps_total": args.steps,
                        "train_loss": loss_val,
                        "avg_loss": running,
                        "best_val": best_val if best_val < float("inf") else None,
                        "tok_s": tok_s,
                        "elapsed_s": elapsed,
                    },
                    indent=2,
                )
            )

        if step % args.eval_every == 0 or step == args.steps:
            tr = eval_loss(train_data, 40)
            va = eval_loss(val_data, 40)
            print(
                f"  eval  train {tr:.4f}  val {va:.4f}  ppl_val {math.exp(min(va, 20)):.2f}",
                flush=True,
            )
            if log:
                log[-1]["eval_train"] = tr
                log[-1]["eval_val"] = va
            # always snapshot so we can resume / inspect mid-run
            save_checkpoint(
                out_dir / "last.pt",
                model, stoi, itos, cfg, step, va, hier, infer_budget, ws, args.optimizer,
                infer_ws=infer,
                tokenizer_path=tok_path,
            )
            if va < best_val:
                best_val = va
                bad_evals = 0
                save_checkpoint(
                    out_dir / "best.pt",
                    model, stoi, itos, cfg, step, va, hier, infer_budget, ws, args.optimizer,
                    infer_ws=infer,
                    tokenizer_path=tok_path,
                )
                print(f"  saved best (val={va:.4f})", flush=True)
                if args.sample_on_best:
                    _print_sample(
                        model, stoi, itos, args.prompt, 160, device, bpe=bpe
                    )
            else:
                bad_evals += 1
                print(f"  no improve ({bad_evals}/{args.patience})", flush=True)
                if args.patience > 0 and bad_evals >= args.patience:
                    print(
                        f"early stop at step {step}: val not improved for "
                        f"{args.patience} evals (best={best_val:.4f})",
                        flush=True,
                    )
                    break

    print("\n=== final sample ===")
    _print_sample(
        model, stoi, itos, args.prompt, args.sample_tokens, device, bpe=bpe
    )

    save_checkpoint(
        out_dir / "last.pt",
        model, stoi, itos, cfg, args.steps, best_val, hier, infer_budget, ws, args.optimizer,
        infer_ws=infer,
        tokenizer_path=tok_path,
    )
    # Always leave a best.pt: if val never beat the resume bar (domain shift,
    # stale best_val), final weights are still the right artifact for export/SFT.
    if not (out_dir / "best.pt").is_file():
        save_checkpoint(
            out_dir / "best.pt",
            model, stoi, itos, cfg, args.steps, best_val, hier, infer_budget, ws, args.optimizer,
            infer_ws=infer,
            tokenizer_path=tok_path,
        )
        print(
            "wrote best.pt from final weights "
            f"(no val improvement over resume bar={best_val:.4f})",
            flush=True,
        )
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    meta = {
        "config": cfg.__dict__,
        "n_params": cfg.n_params(),
        "n_params_torch": sum(p.numel() for p in model.parameters()),
        "infer_budget_bytes": infer_budget,
        "infer_ws_bytes": infer.total,
        "infer_ws_mib": infer.total / (1024 * 1024),
        "train_working_set": ws,
        "optimizer": args.optimizer,
        "backend": "torch",
        "cache": {
            "l1d": hier.l1d_bytes,
            "l2": hier.l2_bytes,
            "l3_instance": hier.l3_bytes,
            "l3_total": hier.l3_total_bytes,
        },
        "best_val": best_val,
        "fits_infer_l3": bool(infer.total <= infer_budget),
        "infer_scope": args.infer_scope,
        "steps": args.steps,
        "tokens": args.steps * tokens_per_step,
        "probe_tok_s": probe_tps,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone. best val={best_val:.4f}  checkpoints in {out_dir}")
    print(
        f"C infer WS: {infer.total / (1024 * 1024):.2f} MiB  vs  "
        f"infer L3 {infer_budget / (1024 * 1024):.1f} MiB  "
        f"({'FITS' if infer.total <= infer_budget else 'OVER'})"
    )


def _enable_checkpointing(model: GPT) -> None:
    """Wrap each block with activation checkpointing.

    Block.forward is ``(x, *, past_kv=None, use_cache=False)``. Training always
    uses past_kv=None / use_cache=False; the wrapper must still accept kwargs so
    GPT.forward can call ``block(x, past_kv=..., use_cache=...)``.
    """
    from torch.utils.checkpoint import checkpoint

    for block in model.transformer.h:
        orig = block.forward

        def make_fwd(fn):
            def fwd(x, *, past_kv=None, use_cache=False):
                if past_kv is not None or use_cache:
                    # KV-cache path is decode-only; do not checkpoint it.
                    return fn(x, past_kv=past_kv, use_cache=use_cache)
                return checkpoint(
                    lambda t: fn(t, past_kv=None, use_cache=False),
                    x,
                    use_reentrant=False,
                )

            return fwd

        block.forward = make_fwd(orig)


def resolve_resumed_best_val(
    resumed_best_val: float,
    resumed_qat: str,
    qat_mode: str,
    *,
    resumed_embed_qat: str = "none",
    embed_qat_mode: str = "none",
) -> float:
    """Early-stop / best.pt bar after resume.

    Fake-quant val CE is not comparable to fp32 (or a different quant mode).
    When the train QAT mode differs from the checkpoint's, start with +inf so
    the first QAT eval can write best.pt and patience does not fire against
    an unreachable fp32 bar.
    Also reset when *leaving* QAT (ternary→none): fp32 bar after quantized
    weights would be meaningless if someone resumes a QAT ckpt without --qat.
    """
    rq = (resumed_qat or "none").lower()
    qm = (qat_mode or "none").lower()
    re = (resumed_embed_qat or "none").lower()
    em = (embed_qat_mode or "none").lower()
    # Any linear or embed quant grid change invalidates the stored best_val.
    if rq != qm or re != em:
        return float("inf")
    try:
        return float(resumed_best_val)
    except (TypeError, ValueError):
        return float("inf")


def _print_sample(model, stoi, itos, prompt, n_tokens, device, bpe=None) -> None:
    model.eval()
    if bpe is not None:
        ids_list = bpe.encode(prompt)
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    else:
        ids = torch.tensor(
            [[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device
        )
    raw = model
    if hasattr(model, "_orig_mod"):
        raw = model._orig_mod
    with torch.no_grad():
        out = raw.generate(ids, max_new_tokens=n_tokens, temperature=0.8, top_k=40)
    print("--- sample ---")
    if bpe is not None:
        print(bpe.decode(out[0]))
    else:
        print(decode(out[0], itos))
    print("--------------")
    model.train()


def save_checkpoint(
    path, model, stoi, itos, cfg, step, val_loss, hier, budget, ws, optimizer,
    infer_ws=None,
    tokenizer_path=None,
) -> None:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    # stable vocab list by id for char mode; BPE uses tokenizer file
    max_id = max(itos.keys()) if itos else -1
    vocab = [itos.get(i, "") for i in range(max_id + 1)]
    # Thread qat modes from model attributes set during train
    qat_tag = getattr(raw, "_ll3m_qat", None)
    embed_qat_tag = getattr(raw, "_ll3m_embed_qat", None)
    payload = {
        "params": raw.state_dict(),
        "vocab": vocab,
        "config": cfg.to_dict(),
        "step": step,
        "val_loss": val_loss,
        "infer_budget": budget,
        "infer_ws_total": infer_ws.total if infer_ws is not None else None,
        "train_ws_total": ws["total"],
        "l3_instance": hier.l3_bytes,
        "optimizer": optimizer,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "qat": qat_tag if qat_tag is not None else "none",
        "embed_qat": embed_qat_tag if embed_qat_tag is not None else "none",
    }
    torch.save(payload, path)


def load_model(path: Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig.from_dict(dict(ckpt["config"]))
    model = GPT(cfg)
    model.load_state_dict(ckpt["params"])
    model.to(device)
    # Match train/serve grid: QAT checkpoints need fake-quant at eval
    qat = str(ckpt.get("qat", "none") or "none")
    embed_qat = str(ckpt.get("embed_qat", "none") or "none")
    if qat != "none":
        from ll3m.qat import enable_qat_on_model

        enable_qat_on_model(
            model,
            num_bits=8 if qat == "int8" else 2,
            ternary=(qat == "ternary"),
        )
        model._ll3m_qat = qat  # type: ignore[attr-defined]
    if embed_qat != "none":
        from ll3m.qat import enable_embed_qat_on_model

        enable_embed_qat_on_model(model, mode=embed_qat)
    model.eval()
    vocab = ckpt["vocab"]
    stoi = {c: i for i, c in enumerate(vocab) if c != ""}
    # re-fill: vocab is list indexed by id
    stoi = {}
    itos = {}
    for i, c in enumerate(vocab):
        itos[i] = c
        if c != "" and c not in stoi:
            stoi[c] = i
    bpe = None
    tp = ckpt.get("tokenizer_path")
    if tp and Path(tp).is_file():
        bpe = BPETokenizer.load(tp)
        stoi, itos = bpe.stoi, bpe.itos
    return model, stoi, itos, bpe


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train max-capability cache-resident GPT")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--val-data", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="microbatch size (keep small on GPU to limit VRAM; raise --grad-accum)",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="gradient accumulation steps (effective batch = batch * accum)",
    )
    p.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=0.45,
        help="max fraction of GPU VRAM this process may use (default 0.45; leave room for desktop)",
    )
    p.add_argument(
        "--qat",
        type=str,
        default="none",
        choices=["none", "int8", "ternary", "binary"],
        help="quantization-aware training: fake-quant Linear weights in forward (STE); "
        "binary={-1,+1} research path (no product C pack default)",
    )
    p.add_argument(
        "--embed-qat",
        type=str,
        default="none",
        choices=["none", "int8", "ternary"],
        help="fake-quant tied wte for embed lookup + LM head (STE); compare int8 vs ternary",
    )
    p.add_argument(
        "--serve-quant",
        type=str,
        default="ternary",
        choices=["ternary", "binary", "int8", "f32"],
        help="C serve quant assumed for L3 infer gate (default ternary product path; "
        "binary=1-bit research estimate)",
    )
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--tokenizer", type=str, default=None, help="path to tokenizer.json")
    p.add_argument("--n-kv-head", type=int, default=None, help="GQA KV heads")
    p.add_argument("--intermediate-size", type=int, default=None, help="SwiGLU mid")
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument(
        "--rope-partial",
        type=float,
        default=1.0,
        help="fraction of head_dim with RoPE (LLaMA-3.1 partial rotary; default 1.0 full)",
    )
    p.add_argument(
        "--logit-softcap",
        type=float,
        default=0.0,
        help="Gemma-style final logit softcap; 0=off, 30 is Gemma default",
    )
    p.add_argument(
        "--z-loss",
        type=float,
        default=1e-4,
        help="PaLM z-loss coeff on logsumexp(logits)^2 (0=off; default 1e-4)",
    )
    p.add_argument(
        "--mtp",
        type=float,
        default=0.1,
        help="multi-token prediction aux loss weight (shared head; 0=off; default 0.1)",
    )
    p.add_argument(
        "--mtp-depth",
        type=int,
        default=2,
        help="how many future tokens to supervise for MTP (default 2 = next+1)",
    )
    p.add_argument(
        "--peri-ln",
        action="store_true",
        help="Gemma2/OLMo2 peri-LN: RMSNorm after each residual add",
    )
    p.add_argument(
        "--layer-scale-init",
        type=float,
        default=0.0,
        help="per-channel residual LayerScale init (0=off; try 1e-2 for deep stacks)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=6,
        help="early-stop after this many evals without val improvement (0=off)",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", type=str, default="lion", choices=["lion", "adamw"])
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str, default="ROMEO:\n")
    p.add_argument("--sample-tokens", type=int, default=500)
    p.add_argument(
        "--infer-scope",
        type=str,
        default="ccd",
        choices=["chip", "ccd"],
        help="L3 scope for C-inference gate (default: one CCD)",
    )
    p.add_argument(
        "--infer-budget-fraction",
        type=float,
        default=0.95,
        help="fraction of infer-scope L3 allowed for C infer WS",
    )
    p.add_argument(
        "--infer-budget-bytes",
        type=int,
        default=None,
        help="override infer L3 gate in bytes",
    )
    p.add_argument(
        "--budget-fraction",
        type=float,
        default=0.90,
        help="(info) train-tensor budget fraction if --enforce-train-budget",
    )
    p.add_argument(
        "--budget-scope",
        type=str,
        default="chip",
        choices=["chip", "ccd"],
        help="(info) train-tensor L3 scope; not the product gate",
    )
    p.add_argument("--budget-bytes", type=int, default=None)
    p.add_argument(
        "--enforce-train-budget",
        action="store_true",
        help="refuse if train tensor WS exceeds train L3 budget (off by default)",
    )
    p.add_argument("--force", action="store_true", help="allow over infer (or train) budget")
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto|cpu|cuda|cuda:N|rocm|hip  (ROCm uses the cuda device API)",
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    p.add_argument("--no-grad-checkpoint", action="store_false", dest="grad_checkpoint")
    p.add_argument("--resume", type=str, default=None, help="resume weights from checkpoint")
    p.add_argument(
        "--sample-on-best",
        action="store_true",
        default=False,
        help="print a short sample whenever val improves (slow)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
