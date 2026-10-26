# Hugging Face survey: what LL3M can learn / steal

**Date:** 2026-07-24  
**Audience:** LL3M product freeze (L23×d576 MQA ~79M, L3 CCD serve ~30.4 MiB, ternary `L3T2` + int8 embeds, T=256, V=2048, C decode + host tools)  
**Method:** Hugging Face API (text-generation downloads + `trendingScore`), model cards, lab blogs, arXiv / tech reports. HF has no public true “last 30 days downloads” field; we use (1) high absolute downloads as a popularity proxy, (2) `trendingScore` as a recent-heat proxy, (3) release blogs for models that exploded this month.  
**Related:** [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) (deeper architecture pass) · [ARXIV_SURVEY_LEARN_STEAL.md](ARXIV_SURVEY_LEARN_STEAL.md) (paper IDs) · [ROADMAP.md](ROADMAP.md) · [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) · [QUANTIZATION.md](QUANTIZATION.md)

**Product filter (non-negotiable):** only ideas that help **language + tool policy under ternary** and **facts outside weights**. Giant MoE, 1M native ctx, vision towers, and NVFP4-as-product are *why frontier does X*, not a shopping list.

---

## 0. Executive ranking (steal first)

| Pri | Steal | Source (HF heat) | LL3M action |
|-----|--------|------------------|-------------|
| **P0** | **True ternary pack that keeps agent/math/coding** (~95% FP16 at ~1.7 b/w) | Prism **Ternary-Bonsai-27B** (trending #3 text-gen) | Stay ternary+int8; measure quality vs fake “2-bit”; never dequant to FP16 on hot path; optional group-scale ablations |
| **P0** | **Agent post-train = persistence + verify + multi-harness**, not more params | Poolside **Laguna S/XS 2.1** (trending #1 / #XS) | Expand tool SFT: verify-before-done, fail-retry, multi-scaffold templates; then GRPO-lite |
| **P0** | **Horizon + domain teachers + on-policy multi-teacher distill** | **Agents-A1** (4B + 35B-A3B trending) | After multi-mix SFT: domain teachers (calc/lookup/no-call) → distill into 79M student under QAT |
| **P0** | **Facts outside residual stream** (Engram / tools / KB) | DeepSeek V4 + Engram; every agent lab | Keep host KB/tools; do not wiki into ternary |
| **P1** | **SFT bootstrap → RL only on low pass-rate tasks** | Laguna, DeepSeek V4 (GRPO domain experts), GLM-5.2 | Roadmap RL: GRPO-lite on short tool tasks with verifiers |
| **P1** | **Thinking on/off budget** (large quality delta at same weights) | Laguna S 2.1, GLM-5.2 effort, Qwen thinking | Product UX: short vs long answer mode; optional CoT budget, not new layers |
| **P1** | **Multi-scaffold tool stability** (same skills, many harnesses) | Laguna, Tencent **Hy3**, Granite tool calling | Train on 2–3 prompt/tool tag variants; eval with `--no-assist` + foreign schema |
| **P1** | **Tiered data / UltraData-style curriculum** | MiniCPM5-1B, SmolLM, Phi, Qwen small | Dense ladder already; keep atomic docs + MinHash + multi-mix |
| **P1** | **Portable int4/NF4 baselines for quality tables** | Qwen3.5/3.6, Gemma 4, gpt-oss | Continue MC vs Qwen family at NF4; **not** NVFP4 product claim |
| **P2** | **MTP / draft for C speed** (not quality) | DeepSeek MTP, GLM MTP, Laguna DFlash, Bonsai DSpark | Optional after quality; C speculative path only if free of WS blowup |
| **P2** | **Softcap / QK stability under low-bit train** | Gemma, Kimi QK-Clip | Keep softcap; tighten if QAT spikes |
| **P2** | **Anti-hallucination + “answer when grounded”** | Hy3 product post-train | Tool SFT: refuse / lookup when unsure; KB hit-rate metric |
| **Skip** | MoE, SWA@T=256, hybrid linear attn for 1M, pure 1-bit (no zero), NVFP4 serve, vision, robot OCR/TTS | Frontier HF catalog | Outside L3 product claim |

**One-line stance (reinforced by July 2026 HF heat):**  
The models people download and trend on are either (a) **small dense utility** (Qwen 0.6B–9B), (b) **agentic post-train at open weights** (Laguna, GLM, Hy3, Agents-A1), or (c) **honest low-bit packs that still reason** (Ternary Bonsai). LL3M is (c)+(b) at CCD scale.

---

## 1. Snapshot: what is popular vs what is hot

### 1.1 Text-generation by absolute downloads (popularity proxy)

Top of HF `pipeline_tag=text-generation&sort=downloads` (sampled 2026-07-24). Embedding/classic BERT models dominate all-pipeline downloads; text-gen is the relevant set.

| Downloads (approx) | Model | Why it matters for us |
|--------------------|-------|------------------------|
| 28M | `Qwen/Qwen3-0.6B` | Edge default; our closest open size-class competitor family |
| 17M | `Qwen/Qwen3-8B` | Quality ceiling reference for mid-small dense |
| 13M | `Qwen/Qwen2.5-1.5B-Instruct` | Still huge; instruct recipe stickiness |
| 10M | `meta-llama/Llama-3.2-1B-Instruct` | Edge instruct baseline |
| 9.4M | `deepseek-ai/DeepSeek-R1` | Reasoning post-train cultural shift |
| 9.3M | `nvidia/Qwen3.6-35B-A3B-NVFP4` | GPU quant demand; **not** our product pack |
| 7.7M | `openai/gpt-oss-20b` | Open OSS reasoning/tool line |
| 6.5M | `Qwen/Qwen3-1.7B` | Dense ladder size ladder |
| 4.5M | `openai/gpt-oss-120b` | Larger OSS sibling |
| 3.7M | `google/gemma-3-1b-it` / Gemma 4 family high | Edge + softcap lineage |
| 3.2M | `deepseek-ai/DeepSeek-V4-Flash` | Current open long-ctx MoE heat |
| 3.2M | `zai-org/GLM-5.2-FP8` | Long-horizon agent flagship |
| 3.1M | `ibm-granite/granite-4.1-8b` | Enterprise tool-calling |
| 2.5M | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Legacy tiny chat; still downloaded |
| 2.2M | `HuggingFaceTB/SmolLM2-135M-Instruct` | Sub-200M instruct exists and ships |

**Popularity lesson:** users still pull **sub-1B and ~1–9B dense** constantly. A 79M L3 product is not “too small to matter” if agent + English work.

### 1.2 Text-generation trending (`trendingScore`) — last ~weeks heat

| Trend | Model | Class | Steal signal |
|------:|-------|-------|--------------|
| 507 | `poolside/Laguna-S-2.1` | 118B-A8B agent coding | Post-train + thinking + trajectories |
| 383 | `upstage/Solar-Open2-250B` | Large open | Watch; low transfer at 79M |
| 349 | `prism-ml/Ternary-Bonsai-27B-gguf` | **Ternary 27B** | **Direct quant validation** |
| 319 | `Nanbeige/Nanbeige4.2-3B` | 3B dense | Small-model recipe |
| 284 | `zai-org/GLM-5.2` | Long-horizon MoE | Effort, IndexShare (skip arch), agent scores |
| 259 | `prism-ml/Bonsai-27B-gguf` | **1-bit** sibling | Compare ternary > binary for product |
| 168 | `Motif-Technologies/Motif-3-Beta` | New lab | Watch architecture blog if published |
| 107 | `fdtn-ai/antares-1b` | **1B** | Edge dense competition |
| 51 | `poolside/Laguna-XS-2.1` | 33B-A3B local agent | Same recipe, smaller active |
| 42 | `InternScience/Agents-A1-4B` | **4B agent** | Horizon > params; multi-teacher |
| 42 | `deepseek-ai/DeepSeek-V4-Pro` | 1.6T-A49B | Engram/mHC/GRPO pipeline |
| 38 | `deepseek-ai/DeepSeek-V4-Flash` | 284B-A13B | Same family, cheaper |
| 45 | `tencent/Hy3` | 295B-A21B | Tool reliability + anti-hallucination |
| 30 | `openbmb/MiniCPM5-1B` | **1B** agent/tools | UltraData tiered training |
| 86 | `google/gemma-4-31B-it` | Dense/MoE mix family | Softcap + edge SKUs |

Also hot outside pure text-gen (ignore for product arch, note ecosystem): OCR (`baidu/Unlimited-OCR`), ASR (`nemotron-3.5-asr`), vision/robot (`MiniCPM-Robot*`), TTS (`Qwen3-TTS`).

### 1.3 What “recent” means this week

Raw `createdAt` dump is mostly noise (finetune clones, LoRAs). **Real recent releases that moved the needle:**

| Window | Release | Source |
|--------|---------|--------|
| ~2026-07-21 | Laguna S 2.1 | [poolside.ai blog](https://poolside.ai/blog/introducing-laguna-s-2-1) |
| ~2026-07-02 | Laguna XS 2.1 | [poolside.ai blog](https://poolside.ai/blog/introducing-laguna-xs-2-1) |
| Mid-2026 | DeepSeek-V4 Flash/Pro | [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) |
| 2026 | GLM-5.2 | [z.ai blog](https://z.ai/blog/glm-5.2), GLM-5 report [arXiv:2602.15763](https://arxiv.org/abs/2602.15763) |
| 2026 | Qwen3.5 / 3.6 family | [qwen.ai blog](https://qwen.ai/blog?id=qwen3.5) |
| 2026 | Ternary/1-bit Bonsai 27B | [Prism whitepaper](https://github.com/PrismML-Eng/Bonsai-demo) |
| 2026-06/07 | Agents-A1 35B + 4B | [arXiv:2606.30616](https://arxiv.org/abs/2606.30616) |
| 2025–26 | gpt-oss-20b/120b | OpenAI open-weight line (still top downloads) |

---

## 2. Model digests (papers / blogs → steal / skip)

### 2.1 Prism ML: Ternary-Bonsai-27B and 1-bit Bonsai (highest quant signal)

| Item | Detail |
|------|--------|
| **HF** | `prism-ml/Ternary-Bonsai-27B-gguf`, `prism-ml/Bonsai-27B-gguf`, MLX 1/2-bit |
| **Refs** | Card + [whitepaper PDF](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf); llama.cpp / MLX forks with native low-bit kernels |
| **Base** | Qwen3.6-27B hybrid-attention backbone (arch unchanged) |
| **Ternary** | Weights in {−1, 0, +1}, FP16 scale per group of **128**, GGUF `Q2_0_g128`; **~1.71 b/w ideal**, ~2.125 b/w deployed until native trit kernels |
| **Claims** | Ternary: **~95% FP16** intelligence (80.49 avg / 15 thinking benches vs IQ2_XXS 72.73); keeps math ~93, coding ~86, agent tools ~74; ~7.2 GB vs ~54 GB FP16 |
| **1-bit sibling** | ~3.9 GB, ~90% FP16, faster tok/s; phone-class |
| **Serve tricks** | Weights **never expanded to FP16** on hot path; 4-bit KV; DSpark draft (~1.34×); hybrid attn helps long ctx (their problem, not ours) |

**Steal**

1. **Ternary is the quality OP; binary is the phone OP.** Matches our product choice (ternary Linear + int8 embed, not pure 1-bit).  
2. **Honest bit-width marketing:** call out fake “2-bit” packs that are really ~2.8 b/w. Document L3T2 true b/w in paper/export.  
3. **End-to-end low-bit including embeds/LM head** (they go further than we do). We already judged embed int8 better than ternary embeds for quality; keep that, but re-check if Bonsai-style absmean group scale recovers embed ternary.  
4. **Custom GEMV that eats packed weights** (we have C path): never dequant-all-to-f32 as default.  
5. **Speculative draft as optional speed** after quality (DSpark/DFlash class), not as training substitute.

**Skip / caution**

- 27B hybrid linear attention + 262K ctx: not the L3 claim.  
- Vision tower: out of scope.  
- Full-model ternary embeds: only if quality holds on our 79M English+tools suite.

**Map to ROADMAP:** P0 quant backlog “train–serve grid lock” + serve quality compare; paper section on ternary retention vs Bonsai-style numbers (cite as concurrent systems evidence).

---

### 2.2 Poolside Laguna S 2.1 and XS 2.1 (highest agent post-train signal)

| Item | Detail |
|------|--------|
| **HF** | `poolside/Laguna-S-2.1`, `poolside/Laguna-XS-2.1`, NVFP4/INT4/GGUF/DFlash |
| **Blog** | [S 2.1](https://poolside.ai/blog/introducing-laguna-s-2-1), [XS 2.1](https://poolside.ai/blog/introducing-laguna-xs-2-1) |
| **Report** | [Laguna M.1 / XS.2 technical report PDF](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf); arXiv family [2605.27605](https://arxiv.org/pdf/2605.27605) |
| **S 2.1 shape** | **118B total / ~8B active**, 48 layers, 256 routed + 1 shared expert, top-10; **12 global + 36 SWA (window 512)**; 1M ctx; vocab ~100K |
| **XS 2.1** | **33B / 3B active**, 40 layers, same family recipe; Mac-local positioning |
| **Train loop** | Same pretrain data XS→S scale-up; **SFT synthetic bootstrap → RL** on tasks with low pass rate; **FP8 RL** on S; **Muon** on XS card |
| **Behavior bet** | Not “more intelligence” alone: **more verification, less taking things for granted, do not declare victory early, persistence** |
| **Thinking** | on/off; max thinking: TB2.1 60.4→**70.2**, DeepSWE 16.5→**40.4** |
| **Eval integrity** | Full trajectories public ([trajectories.poolside.ai](https://trajectories.poolside.ai/)); multi-harness rollouts; reward-hack detection (LLM-as-judge) |
| **Post-train mix** | ~409k envs; large SWE commit/PR/bug; terminal envs; **multi-harness** so skills transfer |
| **Known fails** | Harness overfitting (wrong tool schema memory); nested JSON args; overthinking |

**Steal (this is the goldmine for LL3M agent)**

1. **Train the work style:** verify, backtrack, do not early-exit. Encode in synthetic tool traces (calc/lookup/time) with “check result before final”.  
2. **SFT first for format + basic skill; RL only where pass rate is low** (matches our GRPO-lite plan).  
3. **Multi-harness / multi-template rollouts:** our `<call>` tags + one alternate schema so agent does not overfit assist mode.  
4. **Thinking budget as product flag** (`enable_thinking` / effort), not architecture.  
5. **Publish trajectories** for paper credibility (even if only internal tool battery dumps).  
6. **Longer rollout budgets** during RL than during SFT (patience, not more params).

**Skip**

- MoE 256 experts, SWA interleaved, 1M ctx, Muon-at-all-costs, NVFP4 product pack, DFlash before quality.  
- Copying coding-agent SWE at 79M: wrong domain; steal *behavior training*, not Terminal-Bench chase.

**Map to ROADMAP:** P1 no-assist battery + GRPO-lite; TOOL_CALLING multi-template; paper “agent under L3” related work (Laguna as concurrent open agent recipe).

---

### 2.3 Zhipu GLM-5.2

| Item | Detail |
|------|--------|
| **HF** | `zai-org/GLM-5.2`, FP8 variants |
| **Blog / paper** | [z.ai/blog/glm-5.2](https://z.ai/blog/glm-5.2); GLM-5 [arXiv:2602.15763](https://arxiv.org/abs/2602.15763); IndexShare [arXiv:2603.12201](https://arxiv.org/abs/2603.12201) |
| **Claims** | Solid **1M context**; coding + flexible **thinking effort**; **IndexShare** (reuse sparse-attn indexer every 4 layers, −2.9× FLOPs @ 1M); improved **MTP** acceptance +20% |
| **Bench** | Strong Terminal-Bench / SWE Pro / DeepSWE / tool decathlon class scores vs prior GLM |

**Steal**

- Effort / thinking levels as UX.  
- MTP only as **serve acceleration** research (we already have train MTP aux).  
- Long-horizon **tool + coding RL** narratives for post-train design (not IndexShare).

**Skip**

- IndexShare, sparse attn at 1M, flagship MoE size.

---

### 2.4 DeepSeek-V4 (Flash / Pro)

| Item | Detail |
|------|--------|
| **HF** | `deepseek-ai/DeepSeek-V4-Flash`, `DeepSeek-V4-Pro`, bases |
| **Report** | [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) |
| **Arch** | Hybrid **CSA + HCA** sparse attention; **mHC** residual; **Muon**; MoE (Flash 284B-A13B, Pro 1.6T-A49B); 1M ctx |
| **Post-train** | **Two-stage:** domain experts via SFT+**GRPO** → **on-policy distillation** into one unified model |
| **Prior line** | Engram conditional memory [arXiv:2601.07372](https://arxiv.org/abs/2601.07372); DSA; MLA (see RESEARCH_MODELS_2026) |

**Steal**

1. **Domain experts → on-policy multi-teacher distill** (aligns with Agents-A1).  
2. **GRPO** for verifiable short tools (calc, lookup hit, JSON schema).  
3. **Engram thesis** already productized as host KB.  
4. Optional: mHC-style residual stability if deep thin L23 ever trains poorly (low priority; peri-LN already hurt us once).

**Skip**

- CSA/HCA, 1M, MoE, Muon-required, chasing DeepSWE with 1T models.

---

### 2.5 Qwen3.5 / Qwen3.6 family (popularity kings)

| Item | Detail |
|------|--------|
| **HF** | `Qwen/Qwen3.5-9B`, `Qwen3.6-27B`, `Qwen3.6-35B-A3B`, plus 0.6B–4B dense; many GGUF/NVFP4 remixes trend |
| **Blog** | [Qwen3.5](https://qwen.ai/blog?id=qwen3.5); Qwen3 paper [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) lineage |
| **3.5-9B** | Hybrid layout: **Gated DeltaNet** blocks + gated full attention; SwiGLU; large vocab (~248K); native 262K, extend ~1M; **MTP**; vision early-fusion on some SKUs |
| **RL** | Scaled RL across large agent environment farms |
| **Role for us** | **Quality yardstick** at portable **int4/NF4** on GPU eval host; not architecture target |

**Steal**

- Keep comparing v76 vs Qwen3.5 **0.8B / 4B / 9B** at NF4 (honest gap table).  
- Curriculum + agent RL at *their* scale → our micro version (multi-mix + GRPO-lite).  
- MTP as optional decode speed (already partial).

**Skip**

- GDN hybrid attention for T=256 (KV already tiny with MQA).  
- 201-language expansion (vocab would destroy CCD).  
- NVFP4 as product serve.

---

### 2.6 Google Gemma 4

| Item | Detail |
|------|--------|
| **HF** | `google/gemma-4-31B-it`, 26B-A4B MoE-it, smaller edge SKUs; high downloads |
| **Blog** | [Google Gemma 4 launch](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/); report class [arXiv:2607.02770](https://arxiv.org/abs/2607.02770) (family) |
| **Signals** | Softcap / stability knobs; strong edge + instruct recipes; multimodal SKUs |

**Steal**

- Softcap / logit stability under QAT.  
- Edge positioning narrative (small models as first-class).

**Skip**

- MoE-A4B and vision as product path.

---

### 2.7 OpenAI gpt-oss-20b / 120b

| Item | Detail |
|------|--------|
| **HF** | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` (still multi-million downloads) |
| **Role** | Open reasoning + tool culture; MXFP4 / GPU-centric packs in ecosystem |

**Steal**

- Open eval expectations for tool/reasoning models.  
- Chat/tool template hygiene.

**Skip**

- MXFP4 / GPU-only packs as LL3M serve; 120B scale.

---

### 2.8 InternScience Agents-A1 (4B and 35B-A3B)

| Item | Detail |
|------|--------|
| **HF** | `InternScience/Agents-A1`, `Agents-A1-4B` |
| **Report** | [arXiv:2606.30616](https://arxiv.org/abs/2606.30616) |
| **Thesis** | **Scale the horizon, not the parameters** |
| **Recipe** | (1) full-domain agentic SFT → (2) **domain teacher** models → (3) **multi-teacher multi-domain on-policy distillation** with heterogeneity-aware opts |
| **4B claim** | Strong BrowseComp / GAIA / IFEval class scores vs larger MoEs |

**Steal (almost 1:1 transferable)**

1. Three-stage pipeline maps to: multi-mix SFT → tiny domain specialists (or frozen reward heads) → distill into single 79M QAT student.  
2. Build **verifiers** into the trajectory (knowledge-action-outcome triples).  
3. Evaluate long-horizon *for the domain we care about* (tool chains of length 2–5, not BrowseComp).

**Skip**

- 35B MoE teacher size; web-browse product scope unless host tools expand.

---

### 2.9 Tencent Hy3

| Item | Detail |
|------|--------|
| **HF** | `tencent/Hy3` (295B-A21B, MTP) |
| **Post-train focus** | Tool-call **stability** across scaffolds (variance ≤4% SWE across harnesses); anti-hallucination (answer when grounded); multi-turn intent retention; product-driven issue fix list |

**Steal**

- Explicit **anti-hallucination / grounded answer** SFT lines (tie to `lookup`).  
- Measure tool success **across 2+ scaffolds**.  
- Multi-turn constraint inheritance in synthetic dialogs.

**Skip**

- 295B MoE and product-office frontend chase.

---

### 2.10 MiniCPM5-1B (OpenBMB)

| Item | Detail |
|------|--------|
| **HF** | `openbmb/MiniCPM5-1B` |
| **Recipe** | **UltraData** tiered data management [arXiv:2602.09003](https://arxiv.org/pdf/2602.09003): base → mid → post; strong tools/code/reasoning for 1B |

**Steal**

- Tiered data quality control (we have MinHash + atomic docs; keep raising bar).  
- Proof that **1B-class agent utility** is a real market (supports 79M niche if tools work).

---

### 2.11 IBM Granite 4.1 (8B)

| Item | Detail |
|------|--------|
| **HF** | `ibm-granite/granite-4.1-8b` |
| **Blog** | [HF Granite 4.1](https://huggingface.co/blog/ibm-granite/granite-4-1) |
| **Focus** | SFT + RL for **tool calling**, IF, chat; enterprise RAG/FIM |

**Steal**

- Boring, production tool schemas (OpenAI-style function defs) as optional alternate template.  
- Instruction-following mixes alongside tools (no-call class).

---

### 2.12 Small dense competitors: Antares-1B, Nanbeige4.2-3B, SmolLM, Phi, Ornith

| Model | Signal |
|-------|--------|
| `fdtn-ai/antares-1b` / 350M | Trending 1B; watch card for data/recipe |
| `Nanbeige/Nanbeige4.2-3B` | Trending 3B; dense mid-size |
| SmolLM2/3 | Curriculum + efficient tokens (already in RESEARCH) |
| Phi-4-mini | Reasoning-dense data + SFT/DPO; **explicitly recommends RAG for facts** |
| Ornith GGUF remixes | Community quant demand, not new science |

**Steal:** data density and “facts via RAG/tools” messaging. **Skip:** growing V to match their multilingual vocabs.

---

### 2.13 Other trending (watch-only)

| Model | Note |
|-------|------|
| Solar-Open2-250B | Large open; low transfer |
| Motif-3-Beta | New architecture lab; re-read when report lands |
| ThinkingCap-Qwen3.6-27B | Community thinking merge; recipe remix |
| Kimi-K2.7-Code | Agentic code (Moonshot); same steal as K2: agent factory, not 1T MoE |
| Nemotron NVFP4 / Gemma NVFP4 | GPU train/serve efficiency; **not** CPU L3 product |

---

## 3. Cross-cutting themes (industry consensus mid-2026)

| Theme | Industry does | LL3M does / should |
|-------|---------------|--------------------|
| **Agent > MMLU chase** | Laguna, GLM, Hy3, Agents-A1, Kimi | Product = tools + English under L3 |
| **Test-time compute** | Thinking modes, effort knobs | Optional CoT budget flag |
| **Post-train > arch novelty for mid-size** | Same pretrain data, better SFT/RL (Laguna XS→S) | Freeze shape; invest SFT/RL/data |
| **Multi-teacher / domain then merge** | DeepSeek V4, Agents-A1 | Domain tool teachers → one student |
| **External memory** | Engram, RAG, tools | Host KB v2 + lookup |
| **Honest low-bit** | Ternary Bonsai vs fake 2-bit | L3T2 documentation + quality grid |
| **Eval integrity** | Public trajectories, multi-harness, reward-hack filters | Tool battery dumps; no-assist; multi-template |
| **GPU quant ≠ portable product** | NVFP4, MXFP4 downloads | NF4 for **baselines only** |

---

## 4. Concrete steal backlog (mapped to LL3M work)

### 4.1 Do now (aligns with ROADMAP P0–P1)

| # | Action | Inspiration | Owner artifact |
|---|--------|-------------|----------------|
| 1 | Finish **NF4/int4 MC table** vs Qwen3.5 0.8B→9B | Qwen popularity | `scripts/run_qwen35_family_int4.py`, paper table |
| 2 | **No-assist** agent battery + foreign tool schema variant | Laguna multi-harness, Hy3 scaffolds | `ll3m` agent + TOOL_CALLING |
| 3 | Synthetic traces: **verify before final**, fail-retry, no early victory | Laguna behavior bet | `gen_tool_sft.py` / multi-mix |
| 4 | **GRPO-lite** on verifiable short tools (keep QAT) | DeepSeek GRPO, Laguna RL, GLM | new train path; ROADMAP P1 |
| 5 | Document **true b/w** of L3T2 vs marketing 2-bit | Ternary Bonsai | QUANTIZATION.md + paper |
| 6 | Grounded refuse / lookup-when-unsure lines | Hy3 anti-hallucination | tool SFT mix |

### 4.2 Do next (P2)

| # | Action | Inspiration |
|---|--------|-------------|
| 7 | Domain teachers (calc / lookup / no-call) → on-policy distill into mix student | Agents-A1, DeepSeek V4 |
| 8 | Thinking budget flag in chat CLI | Laguna, GLM effort |
| 9 | Optional C draft / MTP speed path if free of residency regression | DFlash, DSpark, GLM MTP |
| 10 | Softcap/QK audit under longer QAT | Gemma, Kimi |

### 4.3 Explicitly do not steal

| Idea | Why not |
|------|---------|
| MoE at 79M | Routing + expert storage blows CCD story |
| SWA / hybrid linear for 1M | T=256 MQA already tiny KV |
| Pure 1-bit product pack | Bonsai shows ternary wins quality; we need zero state |
| NVFP4 / MXFP4 as serve claim | Product is portable C CPU L3 |
| Vocab 100K–250K | Embeds eat the CCD |
| Vision / robot / TTS | Out of product |
| Muon as required optimizer | Optional experiment only at 79M |
| IndexShare / CSA / HCA | Long-ctx FLOPs; wrong bottleneck |

---

## 5. Citation seeds (for paper / related work)

Use these when expanding `docs/paper/ll3m_l3_resident.bib` (do not claim we trained like them):

| Topic | Seed |
|-------|------|
| Ternary retention at scale | Prism Bonsai whitepaper + HF cards |
| Agentic post-train / trajectories | Poolside Laguna S 2.1 blog + tech report |
| Conditional memory | Engram arXiv:2601.07372 |
| Domain GRPO + distill | DeepSeek-V4 arXiv:2606.19348 |
| Horizon scaling / multi-teacher | Agents-A1 arXiv:2606.30616 |
| IndexShare (related-only) | arXiv:2603.12201 |
| Qwen3 | arXiv:2505.09388 |
| GLM-5 | arXiv:2602.15763 |
| UltraData / MiniCPM | arXiv:2602.09003 |
| BitNet / ternary lineage | existing QUANTIZATION + RESEARCH cites |

---

## 6. Method notes / limits of this pass

1. HF **downloads** are lifetime-ish aggregates, not calendar 30-day. Trending score is the better “last weeks” signal.  
2. Many top downloads are **GGUF remixes** and uncensored merges; we de-emphasized those except where they prove demand for low-bit (Bonsai).  
3. Cards sometimes over-claim benches; we treat Laguna’s public trajectories and Bonsai’s FP16 retention as higher-signal than anonymous leaderboard screenshots.  
4. This doc **complements** [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) (2026-07-17 deep arch pass). Prefer this file for **July 2026 HF heat + steal queue**; prefer RESEARCH for deeper MLA/Mamba/Engram math.

---

## 7. Bottom line

**Steal the factory, not the skyscraper.**

- From **Bonsai**: proof that **true ternary** keeps agent skill; pack honestly; kernels on packed weights.  
- From **Laguna / GLM / Hy3 / Agents-A1 / DeepSeek V4**: **post-train and verifiers and multi-scaffold tools**, thinking budgets, domain→distill, GRPO on short checkable tasks.  
- From **Qwen/Gemma/gpt-oss popularity**: small dense models are the volume market; report quality gaps honestly at portable int4.  
- From **everyone’s MoE/1M/NVFP4**: confirmation that those solve *their* serving economics, not a 32 MiB CCD.

LL3M’s unique claim remains **CCD-gated co-design + host Engram-lite**. The HF wave says the world is moving to **agent post-train + honest low-bit**. That is exactly the product path already on the dense ladder; execute P0/P1 with less architecture thrash.
