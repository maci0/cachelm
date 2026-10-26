# Experiments log (L3 capability)

## P1 eval + readahead (2026-08-03)

### Agent gate instrumentation
- `AgentTurn.gate_events` + battery `gate_stats` (`scripts/eval_agent_battery.py`).
- Suite doc: [AGENT_BATTERY_SUITE.md](AGENT_BATTERY_SUITE.md); tests: `tests/test_agent_gate_stats.py`.
- Product pure_v2: **101/101**, false-call **blocked 27/27**, rewrites **2**.
- Pure dual scores: emission-aware **72/101**, user-visible final **99/101**, freeze pin **90/101**.
- KB on/off lookup: **20/20 → 0/20** (`scripts/kb_onoff_ablation.py`).
- SciQ multi-seed probe (limit 100, seeds 42/43/44 denser-C): **62/60/60** (mean **60.7±1.2**); paper SciQ **66.5** is limit 400 seed 42.
- Status: `outputs/p1-eval-report.md`, `outputs/p1-eval-status.json`.

### Readahead R1/R2/R4 (C path)
- R1 knobs in `c_infer/ll3m_v2.c`: `CLM_W_PREFETCH_AHEAD/HINT/ROW_PREFETCH/NEXT_LAYER_PREFETCH`.
- R2 grid: `scripts/prefetch_grid.sh` — **no keep** (spread &lt;3%, host DRAM WEAK); defaults retained.
- R4: `multi_bench --burst K` single-thread RR; product multi stays **pthread**.
  - N=8 kv=64: pthread agg **~1843** vs RR burst **~216–218**.
- Design + numbers: [READAHEAD_DESIGN.md](READAHEAD_DESIGN.md) §10; `outputs/readahead-r1r2r4-results.json`.
- Golden after knobs: **16/16 PASS**. Paper draft **v0.7.9**.

## C multi-session + hard FIFO (2026-07-27)

- Hard+FIFO single-stream denser-C: **~280–282 tok/s**; soft CCD multi-core **not faster** (~260–270).
- FIFO + quiet core can reach **GOOD (≤5%)** DRAM share of L2 fills on long windows.
- `c_infer/multi_bench`: N-session packing under 30.4 MiB gate.
  - Full T=256: **max N=2**, ~550 tok/s aggregate.
  - Best balanced multi: **N=8, kv_cap=64**, ~**2190** aggregate / ~277 per stream.
  - Peak aggregate: **N=16, kv_cap~32–48**, ~**2300** aggregate / ~145 per stream.
- Sweep: `checkpoints/compare/l3_isolation_bench/n_session_sweep_v2_clean.csv`.
- Paper draft lineage now **v0.7.9** (see [paper/README.md](paper/README.md)).
- Optional RR: `./c_infer/multi_bench ... --burst 16` (fairness; not product agg).

## Dense knowledge ladder (active, 2026-07-21)

Canonical write-up: **[DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md)**.

| ID | Stage | Out / artifact | Status |
|----|-------|----------------|--------|
| **DK-A** | Host KB v2 (aliases, word-bound lookup) | `ll3m/kb_facts.json` (~471 entries / ~1164 keys) | **Done** |
| **DK-B** | Atomic synth 700M + FineWeb 500M + MinHash + short cap | `data/base_en_dense/` | **In flight** (spark1) |
| **DK-C** | Continue ternary QAT +30k | `checkpoints/v76_dense_c/` | Queued |
| **DK-D1/D2** | Fact Q&A SFT + tool SFT (QAT on) | `v76_dense_qa` / `v76_dense_tool` | Queued |
| **DK-V** | Std MC limit=400 vs Qwen3.5 family **weight-only int4** | `checkpoints/compare/*int4*` | **Done** |

**Host:** spark1 `192.168.0.211`.  
**QAT:** ternary Linear + int8 embed locked; assert after each stage.

**Settled MC (limit 400, weight-only int4 Qwen):** C **41.4%** · D **40.5%** · MIX **41.2%** · Qwen 0.8B **54.2%** (−12.8 pp) · 2B **59.4%** · 4B **65.0%** · 9B **69.1%**.  
NF4 baselines deleted (int4 only). Aggregate: `ll3m_vs_qwen35_family_int4.json`.

**Pure model-policy battery (C L3T2, `--no-assist --no-gate`, temp=0):**  
Prior 100/101 denser-C figures used gold inject and are **retired**. Full table: `checkpoints/SHIP_STATUS.json`.

| Ckpt | Total | calc | lookup | no_call | SciQ | notes |
|------|------:|-----:|-------:|--------:|-----:|-------|
| denser C | 42/101 | 0/30 | 0/20 | 24/24 | 66.5 | no tools |
| **pure_v2 (agent)** | **90/101** pin | **30/30** | **20/20** | 15/24 | 57.8 | best tools; product **101/101** gate-on |
| sciq_trade | ~54/101 | ~14/30 | ~14/20 | 9/24 | **67.25** | SciQ hatch |
| v8 pure_v2+dense | 89/101 | 30/30 | 19/20 | 15/24 | 62.3 | SciQ mid |
| v9 more dense | 85/101 | 30/30 | 20/20 | 11/24 | 63.5 | no_call worse |
| d11 best (dense→tools) | 85/101 | 22/30 | 18/20 | 23/24 | 65.0 | denser+tools mid; time 0/5 |
| d11 last | 82/101 | 20/30 | 18/20 | 23/24 | 63.7 | end of d11 |
| d12 calc fix | 82/101 | **30/30** | **20/20** | 7/24 | 60.2 | tools max; no_call overtool; time **5/5** |
| d13 nocall recover | 81/101 | 30/30 | 20/20 | 7/24 | 56.2 | no_call stuck; SciQ worse |
| **d14 gentle from d11** | **89/101** | 26/30 | **20/20** | 17/24 | 61.3 | best denser single pack |
| s2 sciq→tools | 77/101 | 27/30 | 19/20 | 8/24 | 65.8 | tools↑ SciQ near hatch; no_call collapse |

Dense restore from pure_v2 raises SciQ slowly; does not fix gate-off no_call.  
Opposite denser-C → tools: d11 partial, d12 tools max / no_call collapse, d13 stuck,  
d14 gentle from d11 is best denser total (89) but still under pure_v2 tools and SciQ hatch.  
sciq_trade → light tools (s2): calc/lookup recover, SciQ 65.8, no_call overtool (same basin as d12).  
`host_may_inject()` = assist only. pure_v2 golden **16/16**.

**Settled (2026-07-25):** dual ship pure_v2 + sciq_trade. Short QAT from either pole  
(tools-max or SciQ-max) collapses no_call when tool mass rises.

**Dense capability plan (2026-07-26):** [DENSE_CAPABILITY_PLAN.md](DENSE_CAPABILITY_PLAN.md).  
P0 done: product assist=off gate=on; Engram **1022/3057**.  
**2026-07-29 product battery:** pure_v2 gate-on **101/101** (wrong-tool name rewrite: lookup→time; not full assist).  
**Residency:** setcap FIFO OK; quiet pin `--cpu 15 --ccd 1` ~**9.6%** DRAM WEAK; size PASS; measure via `scripts/measure_residency.py`. Doc: [SHIP_FREEZE.md](SHIP_FREEZE.md).  
**P1 done:** `v76_dense_c_p1` SciQ **69.5** (+3 vs dense_c 66.5), macro **~41.4** (flat; BoolQ 56→47.8).  
Paper note: overtrain under QAT can move SciQ without lifting macro or best_val.  
**P2 done:** `v76_dense_c_p2` gate-on 100/101 tools max; pure no_call 6/24; SciQ 65.8 (C+ was 69.5).  
Confirms policy–knowledge Pareto: joint CE from C+ buys tools, spends SciQ and pure no_call.  
**P3a (no ship):** DPO 1k steps: no_call 24/24, calc 4/30, degeneracy.  
**P3b (research):** careful DPO 150 steps, retain 0.55: pure **93/101** (no_call 23/24, calc 27, lookup 16, time 5); gate-on 94.  
**P3c:** from p3b +100 steps lookup-heavy: pure **93/101** (no_call 24/24, lookup 18/20, calc 26, time 3). Total tied; product stays pure_v2. **Stop DPO thrash.**  
**Engram:** **1022** entries / **3057** keys (`scripts/expand_engram_kb.py` wave2: currencies, capitals, CS acronyms, school science). Lookup rank fix: short queries use definitional prefix (`cpu`→full form, not cache).

**GLM5.2 act:** multi-mix / nocall-only / dense thrash did not yield one pack with tools max + SciQ≥66.5 + no_call 24.

**Paper:** `docs/paper/ll3m_l3_resident.tex` **v0.7.9** (int4 tables + gate decomp + KB on/off).  
**Agent suite:** [AGENT_BATTERY_SUITE.md](AGENT_BATTERY_SUITE.md).  
**Readahead:** [READAHEAD_DESIGN.md](READAHEAD_DESIGN.md).  
**Roadmap:** **[ROADMAP.md](ROADMAP.md)**.

## Product line (shape freeze)

| ID | Config | Out-dir | Status |
|----|--------|---------|--------|
| **P1** | **L23 d576 h12/kv1** no peri, LS 1e-4, MTP z-loss | `checkpoints/v75` → ternary / `v76_base_en` | Shape freeze; language → base English v76 |
| P0 (prev) | L18 d320 kv2 peri+LS MTP z-loss | `checkpoints/v2_dense` → `_ternary` | Superseded (left ~17 MiB L3 empty) |

**P1 freeze:** **78,815,712** params, ternary+int8 design WS **~30.22 MiB** / 30.40 MiB gate (peri off, ls 1e-4).
See `docs/TARGET_75M.md`, `docs/PRODUCT_CANDIDATE.md`.

**Live ops:** `checkpoints/v75/progress.json`, `train.log`; waiter log
`checkpoints/v75_ternary/waiter.log`. After QAT:
`scripts/compare_fp32_vs_qat.py` → `quality_compare.json`.

## Ablations (only if P1 plateaus; new out-dir)

| ID | Change | Hypothesis | Ternary WS (est.) |
|----|--------|------------|-------------------|
| A1 | L28 d512 h8/kv1 default mid | Alternate near-max (~78.2M) | ~30.14 MiB |
| A2 | L41 d384 h12/kv1 mid1408 | Depth + fat FFN (~80.5M) | ~30.21 MiB |
| A3 | rope_partial=0.5 | LLaMA-3.1 partial rotary | same as P1 |
| A4 | logit_softcap=30 | Gemma2 final cap | same as P1 |

Do not thrash P1 mid-flight. Compare val NLL on same `mix_val.bin`.

## Quant / BitNet backlog

Tracked in **`AGENTS.md`** (items 1–7): QAT → quality compare → grid lock →
BitLinear → polish → optional A8 / TL GEMV.

## Research follow-ups (after QAT baseline)

Full model digests, arXiv bibliography, and rationale:
**[RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md)** (deep pass 2026-07-17).  
**[HF_SURVEY_LEARN_STEAL.md](HF_SURVEY_LEARN_STEAL.md)** (HF heat + steal queue 2026-07-24).  
**[ARXIV_SURVEY_LEARN_STEAL.md](ARXIV_SURVEY_LEARN_STEAL.md)** (arXiv paper pass + steal queue 2026-07-24).

| ID | Experiment | Source | Metric |
|----|------------|--------|--------|
| R-B | Gold tool-SFT (call + no-call balance) | Kimi, BFCL | `--no-assist-tools` works |
| R-C | Teacher KD into 79M | Ministral, Elastic, Qwen small | Val CE + samples |
| R-D | Curriculum mix stages | SmolLM3, Phi | Capability + tools |
| R-E | Constrained `<call>` decode | SLM FC papers | Format exactness |
| R-H | Host Engram-lite fact table | Engram 2601.07372 | Hit rate, no weight growth |
| R-I | Parallel host multi-tool | Kimi Agent Swarm | Latency |
| R-J | Softcap / QK clamp in QAT | Gemma, Kimi QK-Clip | Spikes, CE |

**Explicit skips** (documented): MoE/MLA/Mamba/DSA at T=256, DiffusionGemma C path,
world knowledge in ternary weights. See research doc decision table.

## Launch P1

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m ll3m.train \
  --arch v2 --tokenizer-mode bpe \
  --n-layer 23 --n-embd 576 --n-head 12 --n-kv-head 1 --block-size 256 \
  --peri-ln --layer-scale-init 1e-2 --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
  --batch-size 4 --grad-accum 16 --gpu-mem-fraction 0.45 \
  --dropout 0.12 --lr 2e-4 --warmup 1000 --steps 50000 \
  --qat none --serve-quant ternary --device rocm --infer-scope ccd \
  --prompt "Once upon a time" \
  --out-dir checkpoints/v75
```

Auto QAT after train finishes:

```bash
bash scripts/run_v75_ternary_qat_when_ready.sh
```

**cap1 (2026-08-01, running):** Resume pure_v2 + `data/cap_mix_v1` (Engram fact Q&A + tool SFT + battery self-distill), QAT ternary+int8, +5k steps → `checkpoints/v76_dense_c_cap1`. Rationale: max capability under freeze without Qwen teacher disk; bake host facts into residual for closed-book while retaining tool policy. Fixed grad-checkpoint `past_kv` kwargs in `ll3m/train.py`.

**Freeze lifted (2026-08-01):** L23×d576 no longer mandatory. L3 gate remains default. Max params T=256 still ~79M (freeze shape); denser path is T=128 ~99M (`scripts/train_dense_t128_max.sh`). Doc: [CAPABILITY_OPEN.md](CAPABILITY_OPEN.md).

**Continuous capability (2026-08-01):** Retrain/FT/RL/data/Engram/shape all allowed. Loop: train → export → battery+SciQ → promote or continue. Queue: cap1 → denser T=128 → mix/KD → GRPO-lite. See [CAPABILITY_OPEN.md](CAPABILITY_OPEN.md).

**cap1 done:** tools gate-on **100/101** (tok sidecar miss only); SciQ **40
**cap1 done (2026-08-01):** tools gate-on 100/101 (tokenizer sidecar only); SciQ 40% (pure_v2 ~58%). Not ship.
**cap2 running:** pure_v2 + cap_mix_v2 (45% mix_train language / 30% tool / 15% fact), +8k QAT, T=256 kept.

**cap2 done:** gate-on 100/101 (tok vs base_en only); SciQ **58.2%** (~pure_v2). Not a clear ship upgrade.
**cap3 running:** pure_v2 + cap_mix_v3 (55% mix_train / 25% tools / fact / p2), +12k QAT, T=256.

**cap3 done:** tools 100/101; SciQ **55.5%** (worse than pure_v2 57.8). No ship.
**cap4 running:** from **dense_c_p1** (SciQ 69.5) + mix 60% language / 20% tools, +8k QAT.
**Battery fix:** tok check uses `data/tokenizer.json` first (not only stale base_en).

**cap4 done (from p1):** tools 100/101 (calc 29/30); SciQ **55.2%** (from 69.5). No ship.
**pure_v2 recheck:** 101/101 with fixed tok check.
**cap5 running:** p1 dense-only continue (no tools) for SciQ twin. Dual ship strategy reaffirmed.

**cap5 done:** dense-only from p1 → SciQ **56.8%** (worse than 69.5), tools **43/101**. Keep p1 as SciQ twin.
**Engram science wave:** 1022→**1056** entries / **3132** keys (school science defs).
**cap6 running:** pure_v2 tools-only + self-distill, +4k QAT.

**cap6 done:** tools-only polish → **99/101** (worse pure_v2); fluency collapse; no ship.
**cap7 running:** pure_v2 + proper multi-mix 50/20/30 (build_multi_mix_sft.py), +10k QAT.

**cap7 done (promote agent):** multi-mix 50/20/30 from pure_v2 → gate-on **101/101**, SciQ **58.8%** (+1 vs pure_v2). Final sample HIP crash; used last.pt.
**cap8 running:** continue multi-mix from cap7, +8k QAT.

**cap8 done:** tools **101/101**; SciQ **53.5%** (worse than cap7 58.8). No promote.
**cap9 running:** from cap7, multi-mix 60/15/25 dense-heavy, +6k low LR.

**cap9 done:** tools **101/101**; SciQ **53.2%** (cap7 58.8). No promote. Multi-mix continues from cap7 regress SciQ.
**cap10 done (NO PROMOTE, early stop ~110.7k):** story-heavy dense-only → SciQ **63.5%** vs p1 **69.5**. Twin stays p1.
**cap11 done (NO PROMOTE):** p1 + SciQ-train atomic → SciQ **58.25%** (worse). Stop short FT thrash off p1 for SciQ twin.
**Lesson:** p1 SciQ peak is fragile; story / atomic / SciQ-train short QAT all regress. Twin stays denser-C p1 until a real denser continue or KD.
**Engram wave3:** **1103** / **3260** (host-only).
**cap12 done (NO PROMOTE):** pure_v2 + science multi-mix → SciQ **63.25%** but battery **81/101** (calc 18/30, lookup 15/20). Tools first; keep cap7.
**cap13 done (NO PROMOTE):** tool-heavy → battery **97/101** (calc 27/30, lookup 19/20), SciQ **58.75**. Better than cap12 tools; still under cap7.
**cap14 relaunched (agent, balanced):** pure_v2 + `cap_mix_v14_balanced` (~38% lang / ~7% fact / ~55% tools), 96k→106k. First auto-launch failed (`torch` had become `2.13.0+cu130`); restored `2.10.0+rocm7.0` and restarted. Promote if gate-on **101** and SciQ **≥ 58.8**.
**L3 capacity search (2026-08-05):** reinvest denser pack → params. Script `scripts/search_l3_capacity.py`, report `docs/L3_CAPACITY_SEARCH.md`, JSON `checkpoints/l3_capacity_search.json`. Ternary+int8 densest ~**80.7M** (+2.4% vs freeze). Binary+int8 densest ~**142M** (+80%).
**Binary scaffold (A):** STE `--qat binary`, export **L3B2**, C load + scalar GEMV. Same freeze pack **11.1 MiB** vs ternary **20.3 MiB**. Product default still L3T2.
**v77_binary_l30 base done (not a ship candidate yet):** L30×d576 mid2304 ≈**142.3M** native binary, 50k on mix_train only. Val CE **1.19**, SciQ@400 **17.75%**. **Fairness:** this is **base only** (no multi-mix tool SFT, no denser SciQ twin stage, no agent polish). Ternary ship (cap7/p1) had those later stages. Do **not** treat 17.75 vs p1 69.5 as binary vs ternary capability final. Next for binary: multi-mix SFT (+ optional denser/SciQ continue) then re-eval.
**cap14 done (NO PROMOTE):** balanced multi-mix → battery **97/101** (calc 28, lookup 18), SciQ **59.25**. Agent stays cap7.
**cap15 stopped early:** pure_v2 multi-mix ~7% of +10k; deprioritized to run fair binary post-base SFT first.
**v77_binary multi-mix SFT done:** SciQ **~19–21.5%** (base 17.75; still ≤ random). Multi-mix val low; battery blocked until L3B2 accepted in `c_engine` (**fixed**).
**v77_binary denser SciQ stage running:** resume SFT `last.pt` @56500 + `cap_mix_v16_binary_sciq` (SciQ train + Engram + light lang/tools), **56.5k→68.5k**, binary QAT. Fairer twin-style step. Script `scripts/train_binary_sciq_stage.sh`.
**train.py fix:** `.bin` data path now loads pre-encoded tokens (was re-encoding as text).

## Handoff (2026-08-10)

Live status and resume instructions: **[HANDOFF.md](../HANDOFF.md)**. v78 binary SciQ denser ~97% at write; product ship still ternary.

## v78 binary C195 (2026-08-10)

L34×d576 mid2880 ~**194.9M** native binary+int8. Ladder **train done**: base val 1.145, SciQ@400 **22.5%** (v77 19.5). C WS **29.56 MiB** fits CCD. Battery was still running at handoff rewrite. **No promote** (ternary ship). Fairness vs dual-ship depth: [BINARY_VS_TERNARY_LADDER.md](BINARY_VS_TERNARY_LADDER.md). Resume: [HANDOFF.md](../HANDOFF.md).

