# Knowledge density in small models

**Snapshot:** 2026-07-21 (ladder implementation note)  
**Question:** How do “small” models compress a lot of general knowledge? How were they trained? What applies to LL3M (~79M ternary, L3T2 ~21 MiB serve pack)?

**Related:** [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) · [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) · [NATIVE_TERNARY_PIPELINE.md](NATIVE_TERNARY_PIPELINE.md) · [TOOL_CALLING.md](TOOL_CALLING.md) · [PRODUCT_CANDIDATE.md](PRODUCT_CANDIDATE.md)

**Implemented ladder (do this, not abstract only):** host KB v2 + aliases, word-boundary lookup, atomic synthetic bias, FineWeb short cap, exact + MinHash near-dedup, full ternary/int8 QAT A→D, MC vs Qwen3.5-0.8B NF4. Full ops: [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md).

**Data-plane densify (research):** [SEMANTIC_COMPRESSOR.md](SEMANTIC_COMPRESSOR.md) §1b argues how claim-preserving train compression raises knowledge *signal per token* under a fixed L3 weight pack (complements Engram; does not expand parametric bit budget). Scaffold: `ll3m/semcomp/`, smoke `data/semcomp_tiny/`.

---

## 1. Thesis for this product

| Layer | Holds | LL3M location |
|-------|--------|----------------|
| **Weights (ternary)** | Language + tool **policy** | L3T2 `.ll3m` |
| **Host tools / tables** | Facts, lookups, live data | `ll3m/tools.py` (`lookup` `_KB`, `wiki`, `calc`, `time`, `note`) |
| **C serve** | L3-resident decode | ≤ ~30.4 MiB CCD gate |

**Do not pack an encyclopedia into ternary weights.** Density fails at this budget. Small models that “know a lot” are almost always **1–4B+** with **GB-scale** payloads and **trillion-token** textbooks/web mixes, not L3-resident packs.

DeepSeek **Engram** (arXiv:2601.07372) formalizes the same split: Transformers lack a native knowledge-lookup primitive and waste compute simulating retrieval; **conditional memory** (O(1) host tables) is a first-class axis next to neural compute. Host `lookup` / `wiki` ≈ Engram-lite.

---

## 2. What “small + knowledgeable” usually means

| People say | Often means | LL3M comparison |
|------------|--------------|-----------------|
| “Small model” | Phone / laptop **1–4B** dense (fp16/bf16 or int4) | We are **~79M** ternary (~**20 MiB** weights) |
| “Lots of knowledge” | Strong **MMLU / school facts / common sense** | Not an offline Wikipedia |
| “Compressed” | **Overtrained** + **curated data** + often **distill** | Not magically more bits per param than physics allows |

### Capacity sketch (order of magnitude)

| Pack | Params (order) | Weight payload (order) | vs L3T2 |
|------|----------------|------------------------|---------|
| **LL3M L3T2** | ~79M ternary | **~21 MiB** | 1× |
| Qwen/Smol class 0.5–1.5B fp16 | ~0.5–1.5B | **~1–3 GiB** | ~50–150× |
| Phi-3-mini 3.8B | 3.8B | **~7+ GiB** fp16-class | hundreds× |
| Phi-2 2.7B | 2.7B | multi-GiB | hundreds× |

A model that “stores lots of facts” has **orders of magnitude more bits** than the product serve pack. Compare methods, not marketing “small.”

---

## 3. How knowledge-dense small models are trained

### 3.1 Shared playbook

```text
1. Filter web for educational value (drop low-value trivia for small capacity)
2. Add synthetic textbooks (explanations, Q&A, step-by-step)
3. Mix code + math (structure + reasoning transfer)
4. Overtrain: tokens >> Chinchilla optimum for that size
5. Curriculum phases: broad language/knowledge → harder reasoning / niche skills
6. Post-train: SFT (+ DPO / RL) for chat and safety
7. Optional: distill from larger teacher; optional: RAG / tools / Engram tables
```

### 3.2 Phi family (Microsoft): “Textbooks Are All You Need”

| Model | Size | Train budget (reported) | Recipe highlights |
|-------|------|-------------------------|-------------------|
| **Phi-1** | ~1.3B (code) | Textbook-quality code + synthetic | Quality ≫ scale for coding |
| **Phi-1.5** | 1.3B | Synthetic “textbook” common sense / science / ToM | Reasoning density for size |
| **Phi-2** | **2.7B** | **~1.4T tokens**; synthetic + edu-filtered web; ~14 days on 96×A100; transfer from Phi-1.5 | Claimed parity with models up to ~25× larger on some benches |
| **Phi-3-mini** | **3.8B** | **~3.3T tokens**; filtered web + synthetic; then SFT/DPO | Phone-deployable; MMLU ~69% class; rivals older much larger chat models on many tests |

**Training methodology (Phi-3 tech report, summary):**

- **Data-optimal regime** for small nets: not pure compute-optimal scaling on fixed messy web.
- **Phase 1:** mostly filtered web → general knowledge + language.
- **Phase 2:** stricter educational subset + synthetic for logical reasoning and niche skills.
- Filter web by **educational level**; remove capacity-wasting trivia (example given in report: day-specific sports scores) so remaining parameters favor **reasoning** and useful knowledge.
- Heavy use of **LLM-filtered web** and **LLM-generated synthetic textbooks**.
- Phi-2: **scaled knowledge transfer** from smaller Phi into larger (warm-start / embed prior knowledge).

**Refs:** [Phi-3 technical report](https://arxiv.org/abs/2404.14219) · [Phi-2 blog](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) · “Textbooks Are All You Need” (Phi-1 line).

### 3.3 SmolLM / open curriculum (Hugging Face)

| Idea | Detail |
|------|--------|
| Sizes | ~135M → 360M → 1.7B → 3B class |
| Style | **Open** multi-stage data mixtures |
| Stages | Web/edu → code/math → high-quality / instruct (exact mix per release) |
| Token budget | Smaller models still **heavily overtrained** |

Reproducible cousin of Phi: educational filters (FineWeb-Edu-style), Cosmopedia-like synthetic textbooks, published curricula. Best reference when we want **transparent** stages rather than closed synthetic stacks.

**Refs:** SmolLM2/3 HF blogs and model cards; see also [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) §SmolLM.

### 3.4 Qwen / Gemma / MiniCPM and peers

| Family | Knowledge levers |
|--------|------------------|
| **Qwen2/2.5/3 small** (0.5B–3B) | Large multilingual + web + code + math; **extreme overtraining**; strong SFT |
| **Gemma-2 2B** | Clean high-quality corpus; often benefits from **shared / distilled** larger-Gemma recipe |
| **MiniCPM** (~1–2B class) | Iterative “model wind tunnel” data A/B; dense bilingual knowledge for size |
| **MobileLLM / on-device SKUs** | Width–depth for latency; still need good data (architecture ≠ free memory) |

### 3.5 Distillation

Many strong small models are **students**:

- Logit / feature KD from a larger teacher.
- Synthetic data **generated by** a larger model (Phi-4 community summaries report large synthetic + rewrite fractions).
- Classic code-model KD into tiny footprints: works better for **skills/format** than arbitrary long-tail facts.

Distillation moves **frequent patterns** well; rare facts still need **capacity** or **retrieval**.

### 3.6 External memory (Engram and tools)

| Approach | Role |
|----------|------|
| **Engram** (DeepSeek) | Conditional O(1) n-gram / hash memory; U-shaped allocation vs MoE compute; offloadable to host DRAM |
| **RAG / wiki API** | Open-world facts at serve time |
| **Closed lookup tables** | Product-friendly fixed KB (`_KB`) |

**Ref:** [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) *Conditional Memory via Scalable Lookup*.

---

## 4. What gets stored well vs poorly

| Mechanism | Stores well | Stores poorly |
|-----------|-------------|---------------|
| Textbook synthetic | Definitions, school science, common sense, explanations | Obscure entities, fresh news |
| Edu-filtered web | Frequent facts, fluent “world model” | Long-tail proper nouns |
| Overtraining | Better use of each parameter | Does not break bit budget |
| Distillation | Teacher’s frequent knowledge + style | Rare facts, imperfectly |
| Tools / Engram / RAG | Arbitrary fact tables if routed | Needs when-to-lookup policy |

Research on **incompressible knowledge probes** (fact questions stratified by obscurity) pushes the same point: **true encyclopedic density is limited by bits in the weights**. Models that “know a lot” often **know common/school knowledge and reason**, and still fail obscure probes.

---

## 5. Datasets used for knowledge-dense small LMs

| Dataset / family | Role for knowledge |
|------------------|--------------------|
| **FineWeb-Edu** / edu classifiers on web | General knowledge from web, high signal |
| **Cosmopedia** / Phi-style synthetic textbooks | Dense explanations and Q&A |
| **Wikipedia / WikiText** | Facts (often better as **retrieval target** than sole pretrain at tiny size) |
| **Code** (Stack, etc.) | Structure; some knowledge transfer |
| **Math** (OpenWebMath and peers) | Reasoning density |
| **xLAM / When2Call / Glaive** | **Tool policy**, not parametric facts |
| **TinyStories** (LL3M default) | **Language fluency**, not world knowledge |

### LL3M current data (baseline)

| Artifact | Content |
|----------|---------|
| Pretrain | **TinyStoriesV2-GPT4** (~1.5 GiB partial → ~**466M** train tokens) + **5% TinyShakespeare** |
| Tokenizer | Domain BPE **V=2048** (`data/tokenizer.json`) |
| Tool SFT | Synthetic When2Call-style mix (`scripts/gen_tool_sft.py` → `data/tool_sft.txt`) |
| Serve knowledge | Host `_KB` + `wiki` + `calc` + `time` |

Manifest: `data/mix_manifest.json`. Download/prepare: `scripts/download_data.sh`, `scripts/prepare_data.py`.

---

## 6. Mapping to LL3M actions

| Their lever | Use at 79M ternary? | How |
|-------------|---------------------|-----|
| Textbook / synthetic Q&A | **Partial yes** | Short fact + explanation episodes; keep TinyStories voice |
| Edu-filtered web | **Limited** | Light spice only unless tokenizer/domain rebuilt |
| Multi-stage curriculum | **Yes** | Story → short Q&A → tool SFT (pipeline stages 0 → 5) |
| Trillion-token overtrain | **Hard / low ROI** | Cost; diminishing returns under ternary + tiny width |
| Match Phi MMLU | **No** | Wrong size and quant regime |
| Distill 0.5B–3B teacher | **Optional later** | Logits or synthetic teacher data; keep QAT grid |
| Pack encyclopedia in weights | **No** | Expand host `_KB` / wiki / tables |
| Tool when-to-call policy | **Yes (product path)** | `gen_tool_sft` + `train_tool_sft.sh` |

### Recommended product path (unchanged)

```text
TinyStories (+ optional light Q&A spice)
    → native ternary pretrain (language + STE grid)
    → tool SFT (policy; keep qat=ternary embed_qat=int8)
    → C serve L3T2
    → host tools for facts (Engram-lite)
```

Optional experiments (not required for freeze):

1. **Micro curriculum:** small synthetic fact+reason blocks (capitals, units, simple science) mixed before or with tool SFT.  
2. **`_KB` growth:** highest ROI for “the model knows X” demos without touching weights.  
3. **KD from open 0.5B–1.7B teacher** after native baseline (see RESEARCH backlog R-C).  
4. **Separate knowledge SKU** if ever needed: multi-GB 0.5B–2B model, **not** the L3 claim product.

### Explicit anti-patterns

| Skip | Why |
|------|-----|
| Wikipedia dump into ternary pretrain | Wastes capacity; broken under V=2048 story BPE; still not encyclopedic |
| Expecting Phi-3-class MMLU at 79M L3T2 | Wrong scale |
| Replacing tools with “just more pretrain” | Engram / density math |
| Tool-only data without no-call balance | Model always emits `<call>` |

---

## 7. Condensed comparison table

| Model / class | Params | Token budget (order) | Knowledge method | Edge relevance to LL3M |
|---------------|--------|----------------------|------------------|------------------------|
| **LL3M product** | ~79M ternary | ~0.5B-class story tokens (+ tool SFT) | Language in weights; facts on host | **This product** |
| SmolLM 135–360M | 0.1–0.4B | Heavy overtrain | Open edu curriculum | Curriculum ideas; still larger payload if fp16 |
| Phi-2 | 2.7B | ~1.4T | Synthetic + edu web + transfer | Data philosophy |
| Phi-3-mini | 3.8B | ~3.3T | Same + phases + align | Data philosophy |
| Qwen small | 0.5–3B | Multi-T class | Mix + overtrain + SFT | Overtrain / multilingual |
| Gemma-2 2B | 2B | Large clean mix | Quality corpus (+ distill family) | Quality bar |
| Engram hybrids | large MoE + memory | large | **External O(1) tables** | **Architecture for facts** |

---

## 8. References (entry points)

| Source | Why |
|--------|-----|
| Phi-3 tech report (arXiv:2404.14219) | Data-optimal small LM; phases; edu filter |
| Phi-2 MSR blog | Textbook + synthetic; knowledge transfer |
| Textbooks Are All You Need (Phi-1 line) | Origin of quality-over-scale recipe |
| SmolLM2/3 HF blogs + datasets | Open curriculum mixes |
| FineWeb-Edu / Cosmopedia | Edu web + synthetic textbooks |
| Engram arXiv:2601.07372 | Conditional memory; facts outside residual stream |
| [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) | Broader 2026 model survey → LL3M actions |
| [TOOL_CALLING.md](TOOL_CALLING.md) | Tool datasets (xLAM, When2Call, Glaive) for **policy** |
| [NATIVE_TERNARY_PIPELINE.md](NATIVE_TERNARY_PIPELINE.md) | Train → export → tool SFT → host KB stages |

---

## 9. One-paragraph summary

Knowledge-dense “small” models (Phi, SmolLM, Qwen-small, Gemma-2B class) earn their reputation by **curating educational and synthetic data**, **overtraining** for trillions of tokens at **multi-billion-parameter** scale, and often **distilling** from larger teachers. They do not violate information limits: they use **far more weight bits** than an L3-resident ternary pack and still lean on **retrieval** for long-tail facts. For LL3M, copy the **curriculum and synthetic-policy** ideas, keep **facts on host tools** (Engram thesis), and treat a true knowledge-heavy model as a **different product** if ever required.

**Under the L3 wall specifically:** you cannot freely grow params, so **data density** (atomic mixes, MinHash, and eventually semantic compression) is how more of a *finite claim curriculum* enters the same pack. That is “more knowledge in the model” only for claims that fit; long-tail stays on host. Theory + experiment contract: [SEMANTIC_COMPRESSOR.md](SEMANTIC_COMPRESSOR.md) §1b.
