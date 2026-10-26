# LL3M documentation index

| Doc | Contents |
|-----|----------|
| [../README.md](../README.md) | Product overview, quick start, layout, claims |
| [OVERVIEW.md](OVERVIEW.md) | Problem statement, data, math sketch, train vs serve |
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | v2 block (RoPE/GQA/SwiGLU), tokenizer, data mix |
| [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) | **Deep pass (2026-07-17):** Mistral, Nemotron, Kimi, GLM, DeepSeek V4, Gemma 4, DiffusionGemma, Qwen, SmolLM, Phi, Llama 4, BitNet + arXiv map → LL3M actions |
| **[HF_SURVEY_LEARN_STEAL.md](HF_SURVEY_LEARN_STEAL.md)** | **HF heat (2026-07-24):** popular downloads + trending (Laguna, Ternary Bonsai, GLM-5.2, Qwen3.5/3.6, Agents-A1, …) papers/blogs → ranked steal/skip |
| **[ARXIV_SURVEY_LEARN_STEAL.md](ARXIV_SURVEY_LEARN_STEAL.md)** | **arXiv pass (2026-07-24):** ternary/QAT, Engram, GRPO/tools, OPD, Agents-A1, sub-100M QAT → ranked steal/skip with paper IDs |
| [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md) | How Phi/SmolLM/Qwen-small pack knowledge; datasets; Engram; mapping to LL3M (do not wiki into ternary) |
| **[SEMANTIC_COMPRESSOR.md](SEMANTIC_COMPRESSOR.md)** | **Research:** train-data semantic compressor; §1b can denser data cram more knowledge under L3? (scaffold `ll3m/semcomp/`) |
| [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) | **Active denser rebuild:** Exp A–D, KB v2, MinHash, atomic mix, QAT lock, spark1 ops, vs Qwen4 MC |
| **[DENSE_CAPABILITY_PLAN.md](DENSE_CAPABILITY_PLAN.md)** | **How to max density (2026-07-26):** stack = weights + Engram + product gate; P0–P4 todos |
| [paper/](paper/) | **Draft paper:** Markdown + LaTeX (`ll3m_l3_resident.tex`) + BibTeX; vs Qwen 0.8B |
| [BASE_ENGLISH.md](BASE_ENGLISH.md) | **Product language goal:** base English over stories; data mix, tokenizer, retrain checklist |
| [NATIVE_TERNARY_PIPELINE.md](NATIVE_TERNARY_PIPELINE.md) | Research → native ternary stages (train, export, tool SFT, Engram-lite tools) |
| [RESEARCH.md](RESEARCH.md) | Earlier lab map + ranked architecture tweaks (still valid; see deep pass for full model survey) |
| [TOOL_CALLING.md](TOOL_CALLING.md) | Tool use design for a tiny L3 model |
| **[AGENT_BATTERY_SUITE.md](AGENT_BATTERY_SUITE.md)** | **Closed 101-case agent suite, dual pure/product, gate_stats** |
| [L3_RESIDENCY.md](L3_RESIDENCY.md) | **No DRAM hot path**: pin, warm, CCD bind, perf gate |
| [READAHEAD_DESIGN.md](READAHEAD_DESIGN.md) | L2 tile / multi-KV readahead; R1 knobs + R4 `--burst` measured |
| [QUANTIZATION.md](QUANTIZATION.md) | int8, **ternary**, QAT, VRAM-safe train |
| [TARGET_75M.md](TARGET_75M.md) | Shape search → L23×d576 freeze |
| [PRODUCT_CANDIDATE.md](PRODUCT_CANDIDATE.md) | Ship candidate summary |
| **[SHIP_FREEZE.md](SHIP_FREEZE.md)** | **Settled dual SKU + residency/setcap measurements (2026-07-29)** |
| **[ROADMAP.md](ROADMAP.md)** | **Phased product roadmap** (P0–P5: measure, ship, RL, quality, quant, publish) |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Active experiment log + research follow-ups |
| **[../HANDOFF.md](../HANDOFF.md)** | **Session handoff / resume** (live train + next actions) |
| **[BINARY_VS_TERNARY_LADDER.md](BINARY_VS_TERNARY_LADDER.md)** | **Fairness:** ternary dual-ship depth vs thin binary ladder; v78 SciQ 22.5 |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Changelog of capability upgrades |
| [../AGENTS.md](../AGENTS.md) | Agent ops, product freeze, quant backlog |
| [../c_infer/README.md](../c_infer/README.md) | Bare-metal C build, run, cachestat |

## Mental model

```
Train (DRAM / ROCm VRAM, any size)
        │  export .ll3m (L3T2 ternary)
        ▼
C inference working set (weights + acts + KV)
        │  must fit ONE L3 CCD (32 MiB here; ~30.4 MiB gate)
        │  mlock + prefault + CCD pin + warm
        ▼
Steady decode: L2 fills from local L3, DRAM ≈ noise
        │  host tools (calc / lookup / wiki / …) for knowledge
        ▼
Agent loop: language + tool policy in weights, facts outside
```

Training tensors need **not** fit L3. Only the C serve path carries the cache claim.

**Research stance (see [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) and [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md)):** parametric weights = language + tool policy under ternary; knowledge = tools / host (Engram thesis).
