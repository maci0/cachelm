#!/usr/bin/env python3
"""Minimal DPO on false_call preference pairs (dense capability P3).

Resume a QAT tool ship pack (e.g. pure_v2). Frozen ref = init weights.
Policy is trained with DPO so preferred plain no_call ranks above rejected
tool spam on the same chitchat prompt. Optional light CE tool retain.

  uv run python scripts/train_pref_false_call.py \\
    --resume checkpoints/v76_dense_c_pure_v2/best.pt \\
    --prefs data/p3_false_call_prefs.jsonl \\
    --out-dir checkpoints/v76_dense_c_p3 \\
    --steps 800 --device cuda

Does not replace ll3m.train; intentional small surface for P3 only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ll3m.config import GPTConfig
from ll3m.model_torch import GPT, Lion
from ll3m.tokenizer import BPETokenizer


def _load_prefs(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty prefs: {path}")
    return rows


def _encode_pair(
    bpe: BPETokenizer,
    prompt: str,
    completion: str,
    block: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (idx, targets, n_comp_tokens) for teacher-forced CE on completion only."""
    full = prompt + completion
    ids = bpe.encode(full)
    if len(ids) < 2:
        ids = ids + [0]
    # crop to block+1 then shift
    if len(ids) > block + 1:
        # keep tail (completion end) so completion tokens stay in window
        ids = ids[-(block + 1) :]
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device)
    # mask: only supervise tokens that belong to completion region
    p_ids = bpe.encode(prompt)
    # number of completion tokens in the uncropped full sequence
    n_comp = max(1, len(bpe.encode(full)) - len(p_ids))
    # after crop, approximate: last n_comp positions of y
    n_comp = min(n_comp, y.numel())
    return x, y, int(n_comp)


def _seq_logprob(
    model: GPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
    n_comp: int,
) -> torch.Tensor:
    """Mean log-prob of last n_comp target tokens (batch size 1)."""
    logits, _ = model(idx.unsqueeze(0), targets=None)
    # logits: [1, T, V]; predict targets at each position
    logp = F.log_softmax(logits[0].float(), dim=-1)
    T = targets.numel()
    start = max(0, T - n_comp)
    token_lp = logp[start:T].gather(1, targets[start:T].unsqueeze(1)).squeeze(1)
    return token_lp.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", type=str, required=True)
    ap.add_argument("--prefs", type=str, default="data/p3_false_call_prefs.jsonl")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="checkpoints/v76_dense_c_p3")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--retain-tool-frac", type=float, default=0.15,
                    help="fraction of steps: CE on a short calc tool gold (retain)")
    ap.add_argument(
        "--early-stop-loss",
        type=float,
        default=0.05,
        help="stop when running DPO loss falls below this (avoid overfit collapse)",
    )
    ap.add_argument(
        "--min-steps",
        type=int,
        default=50,
        help="do not early-stop before this many steps",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    cfg_d = ckpt.get("config") or {}
    if isinstance(cfg_d, GPTConfig):
        cfg = cfg_d
        cfg_save = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
        cfg_save["arch"] = "v2"
    else:
        cfg_save = dict(cfg_d)
        cfg_save.setdefault("arch", "v2")
        fields = {k: cfg_save[k] for k in GPTConfig.__dataclass_fields__ if k in cfg_save}
        cfg = GPTConfig(**fields)

    tok_path = args.tokenizer or ckpt.get("tokenizer_path") or "data/base_en/tokenizer.json"
    bpe = BPETokenizer.load(tok_path)
    prefs = _load_prefs(Path(args.prefs))

    policy = GPT(cfg).to(device)
    policy.load_state_dict(ckpt["params"])
    ref = GPT(cfg).to(device)
    ref.load_state_dict(ckpt["params"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    from ll3m.qat import enable_embed_qat_on_model, enable_qat_on_model

    enable_qat_on_model(policy, num_bits=2, ternary=True)
    enable_embed_qat_on_model(policy, mode="int8")
    enable_qat_on_model(ref, num_bits=2, ternary=True)
    enable_embed_qat_on_model(ref, mode="int8")

    opt = Lion(policy.parameters(), lr=args.lr, weight_decay=0.05)
    block = int(cfg.block_size)
    start_step = int(ckpt.get("step", 0))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # light tool retain golds (full episodes)
    from ll3m.tools import tools_brief

    sys_p = (
        "You are a small helper. Use tools for facts and math. "
        "Answer briefly in plain English.\n" + tools_brief() + "\n\n"
    )
    tool_episodes = [
        sys_p + "User: what is 5+5\nAssistant: <call>calc add 5 5</call>\n"
        "Tool: <result>10</result>\nAssistant: The answer is 10.\n",
        sys_p + "User: what is 10-3\nAssistant: <call>calc sub 10 3</call>\n"
        "Tool: <result>7</result>\nAssistant: The answer is 7.\n",
        sys_p + "User: what is 6 times 7\nAssistant: <call>calc mul 6 7</call>\n"
        "Tool: <result>42</result>\nAssistant: The answer is 42.\n",
        sys_p + "User: what is 100/4\nAssistant: <call>calc div 100 4</call>\n"
        "Tool: <result>25</result>\nAssistant: The answer is 25.\n",
        sys_p + "User: what's 15 plus 5\nAssistant: <call>calc add 15 5</call>\n"
        "Tool: <result>20</result>\nAssistant: The answer is 20.\n",
        sys_p + "User: what is 0+0\nAssistant: <call>calc add 0 0</call>\n"
        "Tool: <result>0</result>\nAssistant: The answer is 0.\n",
        sys_p + "User: What is 100 + 1\nAssistant: <call>calc add 100 1</call>\n"
        "Tool: <result>101</result>\nAssistant: The answer is 101.\n",
        # lookup-heavy (P3b gap: 16/20)
        sys_p + "User: capital of france\nAssistant: <call>lookup capital france</call>\n"
        "Tool: <result>Paris</result>\nAssistant: Paris.\n",
        sys_p + "User: capital of japan\nAssistant: <call>lookup capital japan</call>\n"
        "Tool: <result>Tokyo</result>\nAssistant: Tokyo.\n",
        sys_p + "User: capital of egypt\nAssistant: <call>lookup capital egypt</call>\n"
        "Tool: <result>Cairo</result>\nAssistant: Cairo.\n",
        sys_p + "User: what is pi\nAssistant: <call>lookup pi</call>\n"
        "Tool: <result>3.1415926535</result>\nAssistant: 3.1415926535.\n",
        sys_p + "User: water formula\nAssistant: <call>lookup water formula</call>\n"
        "Tool: <result>H2O</result>\nAssistant: H2O.\n",
        sys_p + "User: boiling water c\nAssistant: <call>lookup boiling water c</call>\n"
        "Tool: <result>100 degrees Celsius at standard pressure</result>\n"
        "Assistant: 100 degrees Celsius at standard pressure.\n",
        sys_p + "User: freezing water c\nAssistant: <call>lookup freezing water c</call>\n"
        "Tool: <result>0 degrees Celsius at standard pressure</result>\n"
        "Assistant: 0 degrees Celsius at standard pressure.\n",
        sys_p + "User: speed of light\nAssistant: <call>lookup speed of light</call>\n"
        "Tool: <result>about 299792458 meters per second</result>\n"
        "Assistant: about 299792458 meters per second.\n",
        sys_p + "User: largest ocean\nAssistant: <call>lookup largest ocean</call>\n"
        "Tool: <result>Pacific Ocean</result>\nAssistant: Pacific Ocean.\n",
        sys_p + "User: days in week\nAssistant: <call>lookup days in week</call>\n"
        "Tool: <result>7</result>\nAssistant: 7.\n",
        sys_p + "User: hours in day\nAssistant: <call>lookup hours in day</call>\n"
        "Tool: <result>24</result>\nAssistant: 24.\n",
        sys_p + "User: what time is it?\nAssistant: <call>time</call>\n"
        "Tool: <result>2026-07-27 12:00 UTC</result>\nAssistant: It is 2026-07-27 12:00 UTC.\n",
    ]

    print(
        f"P3 DPO false_call  resume={args.resume} step={start_step}  "
        f"prefs={len(prefs)}  steps={args.steps}  beta={args.beta}  "
        f"retain={args.retain_tool_frac}  early_stop={args.early_stop_loss}  device={device}",
        flush=True,
    )
    t0 = time.perf_counter()
    running = 0.0
    policy.train()
    done_step = args.steps

    for step in range(1, args.steps + 1):
        # optional tool CE retain
        if random.random() < args.retain_tool_frac:
            ep = random.choice(tool_episodes)
            # split last assistant completion for supervision bias toward full episode
            if "\nAssistant: " in ep:
                # train on full episode as causal LM
                ids = bpe.encode(ep)
                if len(ids) < 2:
                    ids = ids + [0]
                if len(ids) > block + 1:
                    ids = ids[-(block + 1) :]
                x = torch.tensor(ids[:-1], dtype=torch.long, device=device)
                y = torch.tensor(ids[1:], dtype=torch.long, device=device)
                logits, loss_ce = policy(x.unsqueeze(0), targets=y.unsqueeze(0))
                loss = loss_ce if loss_ce is not None else torch.tensor(0.0, device=device)
            else:
                loss = torch.tensor(0.0, device=device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            running = loss.item() if step == 1 else 0.95 * running + 0.05 * loss.item()
        else:
            row = random.choice(prefs)
            prompt = row["prompt"]
            chosen = " " + row["chosen"].lstrip()
            rejected = " " + row["rejected"].lstrip()
            xc, yc, nc = _encode_pair(bpe, prompt, chosen, block, device)
            xr, yr, nr = _encode_pair(bpe, prompt, rejected, block, device)

            # policy
            pi_c = _seq_logprob(policy, xc, yc, nc)
            pi_r = _seq_logprob(policy, xr, yr, nr)
            with torch.no_grad():
                ref_c = _seq_logprob(ref, xc, yc, nc)
                ref_r = _seq_logprob(ref, xr, yr, nr)

            # DPO: -log σ(β [(π_c - π_ref_c) - (π_r - π_ref_r)])
            logits_dpo = args.beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(logits_dpo)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            running = loss.item() if step == 1 else 0.95 * running + 0.05 * loss.item()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(
                f"step {step}/{args.steps}  loss {running:.4f}  "
                f"elapsed {elapsed:.0f}s  abs_step {start_step + step}",
                flush=True,
            )
            (out_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "step": start_step + step,
                        "steps_total": start_step + args.steps,
                        "train_loss": running,
                        "mode": "dpo_false_call",
                        "elapsed_s": elapsed,
                    },
                    indent=2,
                )
                + "\n"
            )

        # early stop: DPO overfit (P3 collapse lesson: loss→0 wiped tools)
        if step >= args.min_steps and running < args.early_stop_loss:
            print(
                f"early stop at step {step}: running loss {running:.4f} "
                f"< {args.early_stop_loss}",
                flush=True,
            )
            done_step = step
            break

    # save checkpoint compatible with train/export
    abs_step = start_step + done_step
    payload = {
        "params": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "config": cfg_save,
        "step": abs_step,
        "val_loss": ckpt.get("val_loss"),
        "best_val": ckpt.get("best_val"),
        "qat": "ternary",
        "embed_qat": "int8",
        "tokenizer_path": str(tok_path),
        "vocab": getattr(bpe, "vocab_size", 2048),
        "train_mode": "dpo_false_call_p3",
        "dpo_steps": done_step,
    }

    last_pt = out_dir / "last.pt"
    best_pt = out_dir / "best.pt"
    torch.save(payload, last_pt)
    torch.save(payload, best_pt)
    print(f"wrote {best_pt} step={abs_step}", flush=True)

    # export ternary pack
    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "scripts/export_weights.py",
            "--checkpoint",
            str(best_pt),
            "--out",
            str(out_dir / "best.ll3m"),
            "--quant",
            "ternary",
        ]
    )
    tok_src = Path(tok_path)
    if tok_src.is_file():
        (out_dir / "best.ll3m.tokenizer.json").write_text(tok_src.read_text(encoding="utf-8"))
    print("DONE P3 DPO", flush=True)


if __name__ == "__main__":
    main()
