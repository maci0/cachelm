# Continuous capability mode (freeze lifted)

**Date:** 2026-08-01  
**Mode:** Keep improving. Retrain, finetune, RL, data, Engram, shape search.  
**Not sacred:** pure_v2, v75 freeze JSON, dual-ship lock, “stop thrash” as a hard rule.

## Soft product defaults (override when metrics win)

| Default | May change if |
|---------|----------------|
| Infer WS ≤ one L3 CCD (~30.4 MiB) | Explicit “abandon L3” decision + new claim |
| L3T2 ternary + int8 embed serve | Better pack under same or new claim |
| Host Engram + tools for long-tail facts | Always preferred for rare facts |
| No isolcpus on this host | Unchanged host policy |

## Improvement loop (repeat forever)

```text
data / Engram / synthetic / harvest
        │
        ▼
train or FT or RL (new out-dir always)
        │
        ▼
export L3T2 → agent battery (gate-on + pure) → SciQ@400 → samples
        │
        ▼
promote ship if better  ──or──  keep exploring
        │
        └────────── next experiment ──►
```

### Levers (use freely)

| Lever | Examples |
|-------|----------|
| **Data** | Dense edu, Engram fact Q&A, tool SFT, self-distill battery wins, multi-mix, web/wiki cards |
| **SFT / continue** | Low LR QAT polish; longer steps; new mix |
| **Shape** | T=256 max ~79M; T=128 denser ~99M; mid/heads search under gate |
| **KD / OPD** | Qwen 0.8B–1.7B teacher when disk allows |
| **RL** | GRPO-lite with **binary verifiers** (calc exact, schema OK, lookup hit); avoid pure preference thrash |
| **Host** | Engram grow, aliases, thin RAG, gate/tool repair |
| **Systems** | C tok/s, multi-session, measure_residency (no isolcpus required) |

### Eval gate (every candidate)

| Metric | Role |
|--------|------|
| Product battery assist off, gate on | Ship agent |
| Pure battery (optional) | Policy stress |
| SciQ + macro @400 | Weight knowledge |
| Samples (story / Q&A / tools) | Fluency |
| Infer WS ≤ gate (if L3 claim on) | Hard for ship packs |

**Promote** any out-dir that wins the trade you care about. Write the winner into `checkpoints/SHIP_STATUS.json` `ship`.

### Anti-patterns (still waste)

- Claiming GOOD DRAM% without evidence  
- Packing encyclopedias into ternary instead of Engram  
- Naive long DPO that kills tools/fluency (P3a lesson)  
- Overwriting the only good pack without a new out-dir  
- Filling disk with abandoned multi-GiB trees (host ~**43 GiB free**)

## Pipeline queue

| # | Stage | Out | Status |
|---|--------|-----|--------|
| 1 | cap1: pure_v2 + Engram/tool mix QAT +5k | `checkpoints/v76_dense_c_cap1` | **running** |
| 2 | Eval cap1 → export → battery + SciQ | same | after (1) |
| 3 | Denser under L3: T=128 ~99M QAT pretrain/FT | `checkpoints/v77_t128_*` | `scripts/train_dense_t128_max.sh` |
| 4 | Longer multi-mix / fact ladder on best | new out-dir | after (2–3) |
| 5 | Teacher KD if disk | new out-dir | optional |
| 6 | GRPO-lite verifiers | new out-dir | after stable SFT baseline |

## Ops

```bash
# live train
pgrep -af ll3m.train
tail -f checkpoints/v76_dense_c_cap1/train.log

# after a run lands best.pt
uv run python scripts/export_weights.py \
  --checkpoint checkpoints/DIR/best.pt --out checkpoints/DIR/best.ll3m --quant ternary
uv run python scripts/eval_agent_battery.py \
  --checkpoint checkpoints/DIR/best.ll3m --no-assist --temperature 0 \
  --out checkpoints/DIR/battery_product_gate_on.json
LIMIT=400 uv run python scripts/run_std_benchmarks.py \
  --checkpoint checkpoints/DIR/best.pt --tasks sciq --limit 400

# denser shape (GPU free)
OUT=checkpoints/v77_t128_l28d576 STEPS=40000 bash scripts/train_dense_t128_max.sh
```

## Disk note

`/home` is ~**99% full**. Prefer one active train out-dir; avoid duplicate `.pt` copies; prune only with explicit user OK.
