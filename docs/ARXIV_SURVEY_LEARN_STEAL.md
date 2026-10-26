# arXiv survey: papers relevant to LL3M (what to steal)

**Date:** 2026-07-24  
**Audience:** LL3M product freeze (L23×d576 MQA ~79M, L3 CCD ~30.4 MiB, ternary `L3T2` + int8 embeds, T=256, V=2048, C CPU decode + host tools)  
**Method:** Targeted arXiv queries (ternary/BitNet/QAT, SLM tools, GRPO/agent RL, Engram/memory, on-policy distill, sub-100M QAT) + export API abstracts + cross-read with [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) and [HF_SURVEY_LEARN_STEAL.md](HF_SURVEY_LEARN_STEAL.md).  
**Not a full arXiv dump:** ~40 high-signal papers, ranked by *transfer to a CCD-gated 79M ternary agent*, not by citation count.

**Product filter:** steal recipes that improve **language + tool policy under ternary** and **facts outside weights**. Skip papers whose only contribution is MoE routing, 1M sparse attention, or GPU-only packs.

---

## 0. Executive ranking (steal first)

| Pri | Steal | Papers | LL3M action |
|-----|--------|--------|-------------|
| **P0** | **Facts outside residual stream** (lookup / N-gram / host KB) | Engram [2601.07372](https://arxiv.org/abs/2601.07372); NGM training-free [2605.16893](https://arxiv.org/abs/2605.16893); LaCy SLM + external source [2602.12005](https://arxiv.org/abs/2602.12005) | Keep/expand host KB + `lookup`; do not wiki into ternary; optional hashed N-gram host table |
| **P0** | **True ternary QAT + train–serve lock** | BitNet [2310.11453](https://arxiv.org/abs/2310.11453), b1.58 [2402.17764](https://arxiv.org/abs/2402.17764), 2B4T [2504.12285](https://arxiv.org/abs/2504.12285); bitnet.cpp [2502.11880](https://arxiv.org/abs/2502.11880); Tequila [2509.23809](https://arxiv.org/abs/2509.23809); Hestia [2601.20745](https://arxiv.org/abs/2601.20745); sub-100M schedule×bit [2605.25966](https://arxiv.org/abs/2605.25966) | Keep QAT; watch dead-zone / STE; match C pack scales; document true b/w |
| **P0** | **When (not) to call tools** + multi-scaffold | When2Call [2504.18851](https://arxiv.org/html/2504.18851); BFCL line; LaCy; Search-not-guess [2604.04651](https://arxiv.org/abs/2604.04651) | Balance no-call ~1:1; train against parametric hallucination (force lookup) |
| **P0** | **SFT → verifiable RL (GRPO-class) on short tools** | RC-GRPO [2602.03025](https://arxiv.org/abs/2602.03025); multi-turn tool RL [2604.02869](https://arxiv.org/abs/2604.02869); synthetic→verifiable [2601.22607](https://arxiv.org/abs/2601.22607); David-GRPO [2601.21699](https://arxiv.org/abs/2601.21699); BitRL [2604.24273](https://arxiv.org/abs/2604.24273) | GRPO-lite under QAT; verifiers for calc/lookup/schema |
| **P1** | **Horizon / domain teachers → multi-teacher OPD** | Agents-A1 [2606.30616](https://arxiv.org/abs/2606.30616); DeepSeek-V4 [2606.19348](https://arxiv.org/abs/2606.19348); OPD survey [2604.00626](https://arxiv.org/abs/2604.00626); CaMOPD [2605.27115](https://arxiv.org/abs/2605.27115) | After multi-mix: domain specialists then on-policy distill into one 79M student |
| **P1** | **Agent work-style / post-train factory** | Laguna M.1/XS.2 [2605.27605](https://arxiv.org/abs/2605.27605); GLM-5 [2602.15763](https://arxiv.org/abs/2602.15763); EnvFactory [2605.18703](https://arxiv.org/abs/2605.18703); TRACE [2604.05336](https://arxiv.org/abs/2604.05336) | Verify-before-done traces; multi-harness; SFT then RL on hard tasks |
| **P1** | **Data curriculum / UltraData / SLM surveys** | UltraData MiniCPM [2602.09003](https://arxiv.org/pdf/2602.09003); SLMs pack a punch [2501.05465](https://arxiv.org/abs/2501.05465); Qwen3 [2505.09388](https://arxiv.org/abs/2505.09388) | Dense ladder + atomic mix; quality over scale |
| **P2** | **PTQ ternary as experiment only** | CAT-Q [2606.26650](https://arxiv.org/abs/2606.26650); post-train ternarize [2510.03267](https://arxiv.org/abs/2510.03267) | Optional cheap ternary port of fp32 teacher; product remains QAT |
| **P2** | **Constrained / grammar decode for `<call>`** | SciDC [2604.06603](https://arxiv.org/abs/2604.06603); structured FC stacks; million-step zero-error [2511.09030](https://arxiv.org/abs/2511.09030) | Strengthen `constrained.py`; optional C mask later |
| **P2** | **Softcap / residual stability** | Gemma softcap lineage; mHC [2512.24880](https://arxiv.org/abs/2512.24880) | Softcap knobs if QAT spikes; mHC only if L23 train dies |
| **Skip** | CSA/HCA/IndexShare/1M MoE, pure 1-bit product, Muon-required, GPU NVFP4 | DeepSeek-V4 arch half, IndexShare [2603.12201](https://arxiv.org/abs/2603.12201), VitaLLM ASIC [2604.27396](https://arxiv.org/abs/2604.27396) | Outside CCD product claim |

**One-line stance:**  
arXiv 2025–2026 agrees with the HF heat: **external memory + agent post-train + honest ternary** beat inventing attention for a sub-100M CPU-resident model.

---

## 1. Theme A — Ternary / extreme quant / edge kernels

### A.1 Canonical BitNet stack (already product baseline)

| ID | Title (short) | Steal |
|----|---------------|-------|
| [2310.11453](https://arxiv.org/abs/2310.11453) | BitNet: Scaling 1-bit Transformers | BitLinear + STE culture |
| [2402.17764](https://arxiv.org/abs/2402.17764) | The Era of 1-bit LLMs (b1.58) | Ternary {-1,0,+1} narrative; absmean scale |
| [2504.12285](https://arxiv.org/abs/2504.12285) | BitNet b1.58 2B4T | Native W1.58 at 2B from scratch |
| [2411.04965](https://arxiv.org/abs/2411.04965) | BitNet a4.8 | Optional later: int8/4 act STE |
| [2502.11880](https://arxiv.org/abs/2502.11880) | bitnet.cpp | **I2_S train–serve match**; TL LUT GEMM ideas for C |

**Action:** Keep **absmean STE** + pack that C loads without silent re-scale. Paper should cite b1.58 + bitnet.cpp as concurrent systems lineage.

### A.2 QAT optimization (dead zones, STE, schedules)

| ID | Title | Steal / caution |
|----|-------|-----------------|
| [2509.23809](https://arxiv.org/abs/2509.23809) | **Tequila**: trapping-free ternary QAT | Dead-zone weights → dynamic bias reactivation; if QAT stalls or many zeros stuck, try Tequila-style bias path (train-only, zero serve cost if annealed) |
| [2601.20745](https://arxiv.org/abs/2601.20745) | **Hestia**: Hessian-guided differentiable QAT | Soften early hard STE; better QAT with fewer tokens vs BitNet/Spectra 100B baselines |
| [2605.25966](https://arxiv.org/abs/2605.25966) | **Schedule × bit-width sub-100M** | Direct size class! Primary hypothesis (INT6 needs different warmdown) **falsified**; **33% warmdown** good across FP16/INT8/INT6 for 15–100M. Steal: do not invent exotic LR schedules for ternary at 79M without a grid |
| ParetoQ / CO-QAT (via Hestia cites) | Scaling laws for QAT compute | Optional: allocate more tokens to QAT polish if CE plateaus |

**Action:** If ternary continue loss is sticky, implement Tequila dead-zone probe before more steps. Log fraction of weights in zero/±1 buckets per layer.

### A.3 PTQ ternary (cheap, usually worse)

| ID | Title | Steal |
|----|-------|-------|
| [2606.26650](https://arxiv.org/abs/2606.26650) | **CAT-Q** ternary PTQ | Learnable modulation + soft…; claims 512 calib samples beat BitNet QAT families on some 1.7–8B ports. **Experiment only** after we have strong QAT baseline |
| [2510.03267](https://arxiv.org/abs/2510.03267) | Post-training ternarization | Same bucket: post-hoc vs QAT |

**Action:** Do **not** replace native/post-hoc QAT with PTQ for product. Optional paper ablation: CAT-Q-style PTQ on v76 dense → L3T2 vs QAT.

### A.4 Hardware / sparsity extras (mostly skip)

| ID | Note |
|----|------|
| Sparse-BitNet [2603.05168](https://arxiv.org/abs/2603.05168) | Ternary + semi-structured sparsity; interesting if L3 still tight after pack |
| VitaLLM [2604.27396](https://arxiv.org/abs/2604.27396) | ASIC dual-core ternary; validates bandwidth-bound decode story, **not** our C path |
| Sherry (ACL 2026, Tequila authors) | 1.25-bit via 3:4 sparse ternary + Arenas; zero-overhead after train. Optional research if we need more density under 30.4 MiB |
| LUT hardware gen [2604.25183](https://arxiv.org/abs/2604.25183) | Affirms: dequant-to-FP16 kills ternary gains (same as our C design rule) |
| BitRL [2604.24273](https://arxiv.org/abs/2604.24273) | **RL on frozen 1.58-bit BitNet** for edge agents + bitnet.cpp. Steal: **post-train RL while keeping quant** (our QAT-on rule) |

---

## 2. Theme B — External memory / knowledge placement

| ID | Title | Steal |
|----|-------|-------|
| [2601.07372](https://arxiv.org/abs/2601.07372) | **Engram**: conditional memory via scalable N-gram lookup | Core thesis: O(1) static lookup as second sparsity axis; U-shaped compute vs memory allocation; frees early layers from memorizing static patterns. **Map to host `lookup`/KB**, not weight-resident Engram (would fight L3) |
| [2605.16893](https://arxiv.org/abs/2605.16893) | **NGM**: training-free plug-and-play N-gram memory | Engram-like **without** training embeddings; host-side hashed tables. Strong fit for Engram-lite v2 |
| [2605.20948](https://arxiv.org/html/2605.20948) | Memory Grafting | Graft memory into small trainable cores; idea only |
| [2601.16531](https://arxiv.org/abs/2601.16531) | Collision-free hot-tier Engram | If host hash KB grows, fix collisions on hot keys |
| [2602.12005](https://arxiv.org/abs/2602.12005) | **LaCy**: what SLMs can/should learn | Capacity bound → external source; studies **when** small models should call out. Aligns with When2Call |
| [2604.04651](https://arxiv.org/abs/2604.04651) | **Search, do not guess** | SLMs **under-call** tools and **parametrically hallucinate**; naive LLM trajectory distill fails because teacher used hidden knowledge. Steal: **force tool use / grounded refuse** in SFT; teacher trajectories must not assume teacher-only facts |
| Physics of LMs knowledge capacity (Allen et al. / Li line) | Scaling laws for facts in params | Do not try to store encyclopedia in 79M ternary |

**Action (concrete):**

1. Expand KB v2 aliases + hit-rate metrics.  
2. SFT lines: “if unsure → lookup” and “if answer in prompt → no tool”.  
3. Optional: host N-gram hash table (NGM-style) for frequent bigrams beyond flat JSON.

---

## 3. Theme C — Tool calling / agents / GRPO

### C.1 Decision and format

| ID | Title | Steal |
|----|-------|-------|
| [2504.18851](https://arxiv.org/html/2504.18851) | **When2Call** | Synthetic when-not-to-call; public models weak; RPO helps. Already on roadmap R-B |
| BFCL (Gorilla) | Function-calling leaderboard | Eval scaffold; irrelevance / multi-step |
| [2501.05465](https://arxiv.org/abs/2501.05465) | SLMs can still pack a punch (survey) | Octopus / TinyAgent / ToolACE: constrained masks, functional tokens, edge FC success rates |
| [2504.19277](https://arxiv.org/html/2504.19277) | SLMs for function calling (empirical) | Small models work for FC with right data; multi-step still hard |
| Octopus v2 [2404.01744](https://arxiv.org/abs/2404.01744) | Functional tokens | Optional: special tokens for tool names (vocab budget careful) |

### C.2 RL for tools (post-SFT)

| ID | Title | Steal |
|----|-------|-------|
| [2602.03025](https://arxiv.org/abs/2602.03025) | **RC-GRPO** | GRPO stalls when group rewards all 0/1; reward-conditioned exploration. **Must-read** before naive GRPO on binary tool success |
| [2604.02869](https://arxiv.org/abs/2604.02869) | Multi-turn RL + iterative reward calibration | Sparse multi-turn credit; step-level calibration |
| [2601.22607](https://arxiv.org/abs/2601.22607) | Synthetic multi-turn → verifiable-reward RL | Factory for tool trajectories + clean rewards |
| [2602.21320](https://arxiv.org/abs/2602.21320) | Tool-R0 zero-data self-play | Self-evolving tool learning; heavy; steal “verifiers first” not full self-play |
| [2601.21699](https://arxiv.org/abs/2601.21699) | **David-GRPO** multi-hop with small agents | Explicitly for **small** models under low compute |
| [2606.16215](https://arxiv.org/abs/2606.16215) | **PACT** privileged trace co-training | Multi-turn tools with privileged traces (dense signal) |
| [2604.05336](https://arxiv.org/abs/2604.05336) | TRACE capability-targeted agent train | Target hard capabilities; not dump all tasks into one RL soup |
| [2605.18703](https://arxiv.org/abs/2605.18703) | EnvFactory executable envs | Scale tool environments; host-side for us (calc/lookup sandboxes already) |
| CoDistill-GRPO (OPD survey cites) | Small+large co-train | Dense KD reward helps **small** GRPO when sparse rewards fail |

**Action:** Roadmap GRPO-lite = **binary verifiers first** (schema OK, calc exact, lookup hit). If advantage vanishes, adopt RC-GRPO reward conditioning or mix dense teacher logprob rewards (CoDistill idea) without critic PPO.

### C.3 Agent systems papers (behavior, not params)

| ID | Title | Steal |
|----|-------|-------|
| [2606.30616](https://arxiv.org/abs/2606.30616) | **Agents-A1**: scale horizon not params | Full-domain SFT → domain teachers → multi-teacher on-policy distill; 4B dense strong. **Blueprint for 79M** |
| [2605.27605](https://arxiv.org/abs/2605.27605) | **Laguna M.1/XS.2** Model Factory | Data automixing, async off-policy agent RL, multi-harness, eval integrity |
| [2602.15763](https://arxiv.org/abs/2602.15763) | **GLM-5** agentic engineering | Async RL infra; ARC capabilities; DSA (skip arch) |
| [2606.19348](https://arxiv.org/abs/2606.19348) | **DeepSeek-V4** | Domain GRPO experts → unified OPD; mHC; hybrid sparse attn (skip last two for L3) |
| [2511.09030](https://arxiv.org/abs/2511.09030) | Million-step zero errors | Decompose tasks; grammar constraints; error as red flag. Steal: short tool chains + hard format checks |

---

## 4. Theme D — Distillation / on-policy transfer

| ID | Title | Steal |
|----|-------|-------|
| [2604.00626](https://arxiv.org/abs/2604.00626) | **Survey of on-policy distillation** | Static SFT imitation fails on long trajectories; OPD = student rollouts + teacher feedback. Production use: Qwen3, DeepSeek-V4, Gemma. **Three-stage recipe:** teacher GRPO → dense OPD bridge → optional student GRPO |
| [2605.27115](https://arxiv.org/abs/2605.27115) | **CaMOPD** multi-teacher OPD | Counteracting domain teachers; gap-based sample selection; domain preserve + general recover |
| Uni-OPD [2605.03677](https://arxiv.org/abs/2605.03677) | Unify OPD dual return | Practical multi-domain merge |
| TGPO [2605.13230](https://arxiv.org/abs/2605.13230) | Teacher-guided policy opt | Dense directional guidance on student rollouts |
| DemoPSD / OPSD leakage papers | Privileged leakage | If self-distill with answers in teacher context, **leakage risk**; prefer external teacher or carefully gated OPSD |
| TinyLLM multi-teacher conflict [2602.01064](https://arxiv.org/abs/2602.01064) | Knowledge conflict | More teachers can **hurt** without conflict handling → CaMOPD / purification |

**Action:** Prefer **external** larger teacher (Qwen 0.8B–4B logits or trajectories) over pure self-distill for 79M. Sequence: multi-mix SFT → (optional) teacher GRPO on tools → **on-policy distill** into QAT student → GRPO-lite if still failing.

---

## 5. Theme E — Architecture / long-context (mostly skip for product)

| ID | Idea | LL3M |
|----|------|------|
| [2512.02556](https://arxiv.org/abs/2512.02556) | DeepSeek DSA | Skip at T=256 |
| [2512.24880](https://arxiv.org/abs/2512.24880) | mHC residual | Optional if deep thin train unstable |
| [2603.12201](https://arxiv.org/abs/2603.12201) | IndexShare | Skip (1M FLOPs) |
| Qwen3 GDN hybrid | Linear + full attn | Skip for T=256 MQA |
| [2502.16982](https://arxiv.org/abs/2502.16982) | Muon / MuonClip | Optional train experiment only |
| MTP (DeepSeek, GLM, Qwen) | Multi-token pred | Keep train aux; optional C draft later |

---

## 6. Theme F — Data / curriculum / SLM product framing

| ID | Steal |
|----|-------|
| [2602.09003](https://arxiv.org/pdf/2602.09003) UltraData | Tiered base → mid → post data management (MiniCPM5) |
| [2505.09388](https://arxiv.org/abs/2505.09388) Qwen3 | Thinking on/off; dense+MoE family; small SKUs as yardsticks |
| [2501.05465](https://arxiv.org/abs/2501.05465) SLM survey | Task-specific SLMs, FC, edge; **not only scale** |
| Phi / SmolLM (see RESEARCH) | Synthetic textbook density; open curricula |

---

## 7. Systems papers adjacent to L3 residency

There is **no** large arXiv literature that claims “entire LLM working set in one CPU L3 CCD.” Closest transferable systems ideas:

| Source | Transfer |
|--------|----------|
| bitnet.cpp + T-SAR [2511.13676](https://arxiv.org/abs/2511.13676) | CPU SIMD ternary; bandwidth critique of naive LUT |
| VitaLLM / LUT hardware papers | Bandwidth-bound decode; packed weights, no dequant |
| Serving / guided decode papers | Schema-constrained generation for tools |
| **This product’s novelty** | CCD gate + mlock/warm + ternary pack co-design (paper claim, not arXiv prior art) |

**Paper implication:** related work should cite BitNet/bitnet.cpp/Engram/agent-SLM papers, then **position CCD-gated co-design** as the systems contribution.

---

## 8. Concrete steal backlog (merged with ROADMAP)

### Do now

| # | Action | Paper drivers |
|---|--------|---------------|
| 1 | When2Call-style **no-call** balance + anti-hallucination (“search, don’t guess”) | 2504.18851, 2604.04651, LaCy |
| 2 | Verifiers for calc/lookup/schema before GRPO | 2601.22607, RC-GRPO |
| 3 | If GRPO stalls on all-0/all-1 groups → **RC-GRPO** or dense teacher reward | 2602.03025, OPD survey |
| 4 | Tequila/Hestia dead-zone audit on QAT runs | 2509.23809, 2601.20745 |
| 5 | Sub-100M: prefer **standard warmdown (~33%)** over exotic schedules | 2605.25966 |
| 6 | Document true L3T2 b/w + train–serve lock (I2_S) | bitnet.cpp, Bonsai (HF) |

### Do next

| # | Action | Paper drivers |
|---|--------|---------------|
| 7 | Domain teachers → multi-teacher **OPD** into mix student (keep QAT) | Agents-A1, DeepSeek-V4, CaMOPD, OPD survey |
| 8 | Host NGM-style hashed N-gram table (training-free) | Engram, NGM |
| 9 | Constrained decode harden for `<call>` | SciDC, BFCL stacks, 2511.09030 |
| 10 | Optional CAT-Q PTQ ablation for paper | 2606.26650 |

### Do not

| Idea | Why |
|------|-----|
| Implement CSA/HCA/IndexShare | Wrong bottleneck |
| Weight-resident Engram in the 30 MiB WS | Host-side only |
| Replace QAT with PTQ for ship | Quality risk |
| Full critic PPO | GRPO-lite + verifiers enough |
| Muon as required | AdamW/Lion fine at 79M |
| Vocab growth for functional tokens without budget check | CCD embeds |

---

## 9. Suggested paper / bib seeds (IDs only)

```
BitNet 2310.11453
BitNet-b1.58 2402.17764
BitNet-2B4T 2504.12285
bitnet.cpp 2502.11880
Tequila 2509.23809
Hestia 2601.20745
CAT-Q 2606.26650
sub100M-QAT-schedule 2605.25966
Engram 2601.07372
NGM 2605.16893
When2Call 2504.18851
Search-not-guess 2604.04651
LaCy 2602.12005
RC-GRPO 2602.03025
multi-turn-tool-RL 2604.02869
OPD-survey 2604.00626
Agents-A1 2606.30616
DeepSeek-V4 2606.19348
GLM-5 2602.15763
Laguna-tech 2605.27605
Qwen3 2505.09388
SLM-survey 2501.05465
UltraData 2602.09003
mHC 2512.24880
DSA 2512.02556
```

Add full BibTeX when paper v0.5 is cut (`docs/paper/ll3m_l3_resident.bib`).

---

## 10. Relation to other docs

| Doc | Role |
|-----|------|
| [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) | Model-family digests + older arXiv map |
| [HF_SURVEY_LEARN_STEAL.md](HF_SURVEY_LEARN_STEAL.md) | July 2026 HF downloads/trending + blogs |
| **This file** | **Paper-first** steal queue with IDs and actions |
| [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md) | Capacity math / datasets |
| [ROADMAP.md](ROADMAP.md) | Execution phases |
| [QUANTIZATION.md](QUANTIZATION.md) | Our QAT/pack details |

---

## 11. Bottom line

arXiv does not invent a second product for us. It **confirms and refines** the one we have:

1. **Ternary works** if QAT is careful (Tequila/Hestia/sub-100M schedule) and serve matches train (bitnet.cpp).  
2. **Knowledge must leave the weights** (Engram/NGM/LaCy/search-not-guess).  
3. **Agents are post-train systems** (When2Call, GRPO variants, OPD, Agents-A1, Laguna factory), not bigger residual streams.  
4. **Long-context MoE papers** are related work for the industry, not recipes for a 32 MiB CCD.

Steal the **factories and verifiers**; keep the freeze.
