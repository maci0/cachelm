# Product candidate: max capability under one L3 CCD

## Language goal (2026-07-20)

**Primary:** proper **base English** + **compressed knowledge density** under L3T2.  
**Not primary:** TinyStories-style children’s narrative.

### Curriculum (do not reorder)

1. **Stage 1 — base:** basic English + dense edu/KB-style pretrain (`v76_base_en`).  
2. **Stage 1b — denser ladder (active):** atomic mix + KB v2 + continue QAT + fact/tool SFT on spark1 → `v76_dense_*`. See **[DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md)**.  
3. **Stage 2 — tool SFT:** When2Call mix, keep ternary QAT (included as ladder Exp D2).  
4. **Serve:** L3T2 + host tools for long-tail / live facts.

Current `v75*` checkpoints were TinyStories-trained (**story-skewed**). Details: **[BASE_ENGLISH.md](BASE_ENGLISH.md)**.

Weights: language + compact common knowledge + tool **policy**. Long-tail facts: **host tools** (Engram-lite KB v2 + wiki/calc).

**Primary agent ship:** `checkpoints/v76_dense_c_pure_v2` with **assist off, gate on** (product; **101/101** after 2026-07-29 wrong-tool gate rewrite).  
**SciQ peak weights:** `checkpoints/v76_dense_c_p1` (SciQ **69.5**, macro ~41.4).  
**Freeze write-up:** [SHIP_FREEZE.md](SHIP_FREEZE.md).  
**SciQ SFT twin:** `checkpoints/v76_dense_c_sciq_trade` (SciQ 67.25).  
**Host density:** `ll3m/kb_facts.json` Engram-lite (**1022** entries / **3057** keys; wave2 2026-07-30).  
**Plan:** [DENSE_CAPABILITY_PLAN.md](DENSE_CAPABILITY_PLAN.md).  
**Paper draft:** [paper/LL3M_L3_Resident_Ternary_SLM.md](paper/LL3M_L3_Resident_Ternary_SLM.md) **v0.7.9**.  
**Agent suite + gate stats:** [AGENT_BATTERY_SUITE.md](AGENT_BATTERY_SUITE.md).  
See durable table: `checkpoints/SHIP_STATUS.json`.  
Prior denser-C 100/101 pure claims used gold inject and are **invalid**.

| Pack | Gate-off battery | calc / lookup / no_call | SciQ | Role |
|------|------------------|-------------------------|------|------|
| denser C | 42/101 | 0 / 0 / 24* | 66.5 | MC baseline (macro 41.4) |
| **pure_v2** | **90/101** pure pin / **101/101** product | **30 / 20 / 15** pure pin | 57.8 | **agent ship** (gate-on product; see gate decomp) |
| sciq_trade | ~54/101 | ~14 / 14 / 9 | 67.25 | SciQ SFT twin |
| **dense_c_p1 (C+)** | (no tool SFT) | — | **69.5** | **SciQ peak weights** (macro flat 41.4; BoolQ−8) |
| dense_c_p2 (joint) | 80 pure / 100 product | 30 / 20 / 6 pure | 65.8 | tools max + time 5/5; pure no_call collapsed |
| dense_c_p3b (DPO) | **93 pure** / 94 product | 27 / 16 / **23** pure | — | best pure balance; product still pure_v2 |
| dense_c_p3c (DPO) | **93 pure** / 93 product | 26 / **18** / **24** pure | — | max pure no_call + better lookup; time down |
| v8 (pure_v2+dense) | 89/101 | 30 / 19 / 15 | 62.3 | mid SciQ, tools held |
| v9 (more dense) | 85/101 | 30 / 20 / 11 | 63.5 | no_call regressed |
| d14 (denser→gentle tools) | 89/101 | 26 / 20 / 17 | 61.3 | best denser single pack |

\* denser no_call is mostly no-`<call>` garbage.  
Pack L3T2 ~20.3 MiB; pure_v2 golden **16/16**; WS ~30.22/30.40.

### C serve throughput (reference 9950X CCD)

| Mode | Number |
|------|--------|
| Hard+FIFO single-stream | **~280–282 tok/s** |
| Soft single-stream | ~260–270 (not preferred) |
| Multi N=2 full T=256 | ~**550** aggregate |
| Multi N=8 kv=64–128 | ~**2100–2200** aggregate |
| Multi N=16 short KV | ~**2300** aggregate peak |

Tool: `c_infer/multi_bench` (pthread product; optional `--burst K` RR research).  
Details: [L3_RESIDENCY.md](L3_RESIDENCY.md), [READAHEAD_DESIGN.md](READAHEAD_DESIGN.md), `checkpoints/compare/l3_isolation_bench/N_SESSION_MAX.md`.

### Agent gate decomposition (instrumented, pure_v2)

Product **101/101** is model emission × host gate (assist off, gate on):

| Stat | Value |
|------|------:|
| Protocol cases | 85 |
| Model emitted `<call>` | 85/85 |
| Gold needs tool | 58 (all emitted; 0 silent) |
| False calls blocked | **27/27** |
| Tool name rewrites | **2** |
| Pure emission-aware | **72/101** |
| Pure user-visible final | **99/101** |
| Freeze pure pin (stream) | **90/101** |

KB on/off (lookup n=20): **20/20 → 0/20** empty table (host memory, not weights).  
Artifacts: `battery_*_gatestats.json`, `kb_onoff_ablation.json` under `checkpoints/v76_dense_c_pure_v2/`.

Host: `host_may_inject()` = **assist only**. Dense restores from pure_v2 raise SciQ slowly (57.8→63.5) without closing gate-off no_call.  
Denser→tools ladder (d11–d14): d12 maxes calc/lookup/time but collapses no_call; d14 is best denser total (89) still under pure_v2 tools and SciQ hatch. Stop d-series thrash.  
**Roadmap:** [ROADMAP.md](ROADMAP.md).

## Freeze (v75 → v75_ternary)

| Field | Value |
|-------|--------|
| Out-dir | `checkpoints/v75` then `checkpoints/v75_ternary` |
| Arch | v2 dense decoder |
| Shape | **L23 × d576**, heads **12/1** (MQA), T=256, V=2048 BPE |
| intermediate | **1536** (default ≈8/3 d, 64-aligned) |
| Extras | **no peri-LN**, LayerScale **`1e-4`**, MTP `0.1`/depth2, z-loss `1e-4` |
| Params | **78,815,712** |
| Serve quant | **ternary Linears (2-bit) + int8 embeds** (C default product pack) |
| Compact infer WS | **~30.22 MiB** (weights ~21.20 + acts ~9.01) |
| L3 gate | 95% of one CCD = **30.40 MiB** |
| Gate result | **FITS** (**99.4%** fill) |

Train note: peri-LN stuck this width/depth at val CE ~5.9 in 300-step ablations;
LayerScale `1e-4` alone reached ~3.2. Failed peri run archived as
`checkpoints/v75_failed_peri`.

Chosen by exhaustive max-params search under the product gate with standard
heads `{8,12,16}`, practical `head_dim ∈ {32,40,48,64}`, and default SwiGLU mid.
See `docs/TARGET_75M.md`. Absolute exotic max is ~80.9M (odd width/mid); this
shape is the practical optimum.

## Train recipe

### A. Two-phase (legacy v75 path)

1. **Phase 1 (capability):** `--qat none`, lr `3e-4`, warmup 2k, 50k steps, Lion, dropout 0.1, patience 0, GPT-4 mix (~466M tok)  
2. **Phase 2 (serve):** `--qat ternary --embed-qat int8`, resume best, lr `5e-5`, +6–8k steps  

Script: `scripts/train_dense_ternary.sh` / `scripts/run_v75_ternary_qat_when_ready.sh`.

### B. Native ternary from step 0 (preferred product path)

Full research map: **[NATIVE_TERNARY_PIPELINE.md](NATIVE_TERNARY_PIPELINE.md)**.

- **Single run:** `--qat ternary --embed-qat int8` from init (no fp32 phase)  
- Train knobs from research: z-loss, MTP, LayerScale, warmup **3k**, softcap **off** (C parity)  
- Out-dir: `checkpoints/v75_native_ternary`  
- Scripts: `train_v75_native_ternary.sh`, waiter `run_v75_native_ternary_when_ready.sh`  
- After train: export L3T2; compare vs `v75` + `v75_ternary`; **tool SFT** (When2Call mix, QAT kept on)  
- Host **Engram-lite** lookup KB in `ll3m/tools.py` (facts outside weights)

Keep ternary (+ int8 emb) on for any later fine-tunes of this lineage.

## Research: kept vs deferred (L3 / portable C)

### Kept (in this candidate)

| Technique | Lab lineage | Why |
|-----------|-------------|-----|
| RMSNorm, RoPE, SwiGLU, MQA, no bias | LLaMA / Qwen dense | Quality per param; MQA frees KV for params |
| QK-Norm | Qwen3 / Gemma3 / OLMo | Train stability |
| Peri-LN | Gemma2 / OLMo2 | Deep residual stability |
| LayerScale | deep nets / nanochat-adjacent | Helps L23 |
| Shared-head MTP | DeepSeek-style signal | Train-only richer CE |
| z-loss | PaLM / OLMo | Logit stability |
| Tied embeds | MobileLLM / small LMs | L3 density |
| Wide-deep L23×d576 | L3 fill search | Max params under gate |
| Domain BPE V=2048 | SLM practice | English density without fat embeds |
| GPT-4 TinyStories mix | early v75 path | **Legacy data**; superseded by base-English goal ([BASE_ENGLISH.md](BASE_ENGLISH.md)) |
| Post-base ternary QAT + int8 embeds | Edge quant ablations | Match C pack; embeds stay int8 |

### Deferred (with reason)

| Technique | Why not for this product |
|-----------|---------------------------|
| MLA / CSA / HCA / SWA / cross-layer KV | Long-context (4k–1M); little gain at T=256; C cost |
| MoE | Residency + C routing complexity |
| Full DeepSeek MTP modules | Extra weights; shared-head MTP first |
| mHC (V4 residual streams) | Memory + C complexity; try if L23 plateaus later |
| Engram **inside** weights | Host tools / hashed facts are the Engram for L3 (arXiv:2601.07372) |
| Hybrid Mamba / GDN linear attn | Long-ctx throughput; useless win at T=256 MQA |
| Diffusion LM (DiffusionGemma) | Different C decode graph; quality/facts trade |
| GQA kv=2 at this size | Inflates acts/KV; caps params ~8–9M lower under gate |
| Exotic L37×d408 mid1472 (~80.9M) | Odd width/hd; prefer standard GEMM dims |

**Full survey:** [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md).  
**Backlog:** `AGENTS.md` quant 1–7 + research R-B…R-J.

## Data

- Primary: TinyStoriesV2-GPT4 (~1.5 GiB partial + official valid)  
- Story-level split, quality filters, cross-dedup, Shakespeare spice ~5% train only  
- Pre-encoded `data/mix_*.bin`, `data/tokenizer.json` V=2048, `mix_manifest.json`  
- Current rebuild: **~466M train tokens**, ~6.0M val tokens (~1.95M stories)  

## Comparison baseline

Older `checkpoints/v2_dense` was **L18×d320** (~20.8M, ~13.1 MiB ternary). This
candidate fills the CCD: **~78.8M params / ~30.3 MiB**. Primary metric: multi-k-step
val NLL on the new val + sample quality under ternary QAT.

## Train results (recorded)

| Phase | Out-dir | Steps | best_val (CE) | Notes |
|-------|---------|-------|---------------|-------|
| Base fp32 | `checkpoints/v75` | (running) | — | L23 d576 product freeze |
| Ternary+int8 emb QAT | `checkpoints/v75_ternary` | — | — | after phase 1 best |

## Open tool SFT

Separate stage after ternary base (see [TOOL_CALLING.md](TOOL_CALLING.md)).
Host agent loop exists (`ll3m/agent.py`); gold tool SFT without assist is
research item **R-B**. Knowledge stays **outside** weights (Engram thesis).
