# Deep research: popular models + arXiv → LL3M

**Date:** 2026-07-17 (deep pass)  
**Audience:** product freeze L23×d576 MQA (~79M), L3 CCD serve (~30 MiB gate), ternary L3T2, BPE V=2048, T=256, C CPU decode + host tools  
**Scope:** Mistral/Ministral, Nemotron, Kimi K2/K2.5/(K3 status), GLM-5/5.2, DeepSeek V3/V4, Gemma 4, DiffusionGemma, Qwen3/3.5, SmolLM, Phi, Llama 4, BitNet/ternary, plus arXiv systems papers  
**Method:** HF model cards, lab blogs, arXiv tech reports, architecture galleries; filter to *transferable* ideas for a ~80M L3-resident ternary product  

**Goal:** document every finding that could help. Giant MoE / 1M context is *why frontier does X*, not a shopping list.

**July 2026 HF heat refresh:** [HF_SURVEY_LEARN_STEAL.md](HF_SURVEY_LEARN_STEAL.md) (Laguna S/XS 2.1, Ternary Bonsai 27B, GLM-5.2, Agents-A1, Qwen3.6, Hy3, … ranked steal queue). Prefer that file for current downloads/trending; prefer this file for deeper arch notes.

**July 2026 arXiv paper pass:** [ARXIV_SURVEY_LEARN_STEAL.md](ARXIV_SURVEY_LEARN_STEAL.md) (ternary QAT Tequila/Hestia/CAT-Q, Engram/NGM, GRPO/OPD, Agents-A1, sub-100M schedules, … with arXiv IDs).

---

## 0. Executive summary

| Priority | Idea | Source family | Action for LL3M |
|----------|------|---------------|-----------------|
| **P0** | Train on the **same quant grid** as serve (ternary STE + int8 emb) | BitNet b1.58, bitnet.cpp, Tequila, TernaryLM | **Native from step 0** (`v75_native_ternary`) + post-hoc QAT baseline (`v75_ternary`) |
| **P0** | **Agentic skill factory**: synthetic tool traces + verifiers, not raw knowledge | Kimi K2, Nemotron, GLM-5, SLM tool papers | Expand `gen_tool_sft` + when2call balance |
| **P0** | **External knowledge**, not parametric facts | Engram thesis, GLM5.2 retrieval-lift, tool density math | Host tools / lookup; weights = language + policy |
| **P1** | **Distill / prune / cascade** from larger teacher into dense student | Ministral 3, Nemotron Elastic, Phi, Qwen small SKUs | Optional after QAT: teacher logits KD |
| **P1** | **Data curriculum** multi-stage mix (web → code/math → quality/tools) | SmolLM2/3, Qwen3, Phi | Improve `mix_*` beyond stories |
| **P1** | **Short-call tool protocol** + constrained decode | Kimi templates, Octopus/SimpleTool, BFCL | Keep agent loop; add constrained `<call>` later |
| **P1** | Softcap / QK stability during QAT | Gemma softcap, Kimi QK-Clip | You have softcap; tighten if QAT spikes |
| **P2** | Hybrid **linear attention / Mamba** for long seq | Nemotron 3, Qwen3-Next/3.5 GDN | Only if T ≫ 256; L3 KV already tiny (MQA) |
| **P2** | MLA / DSA / IndexShare for KV/compute | DeepSeek V3–V4, GLM-5.2, Kimi | Overkill at T=256 MQA |
| **P2** | Speculative decode / MTP | DeepSeek MTP, GLM-5.2 MTP | Optional C speed path |
| **P2** | **Engram-style** O(1) fact table outside weights | DeepSeek Engram arXiv:2601.07372 | Conceptual twin of host `lookup`/wiki; research host-side hash tables |
| **P3** | Diffusion LM (block denoise) | DiffusionGemma, LLaDA, BD3-LM | Different paradigm; research only |
| **P3** | Muon / MuonClip optimizer | Kimi K2, arXiv:2502.16982 | Optional train experiment; AdamW fine for 79M |
| **Skip** | Ultra-sparse 1T MoE, 256K–1M native ctx, vision encoders, NVFP4 GPU pretrain | Frontier HF | Outside L3 product claim |

**One-line product stance (reinforced by Engram + every agent lab):**  
Parametric weights = **language + tool policy** under ternary. Knowledge = **tools / host / external memory**. Architecture = keep **dense modern Transformer** unless T or memory story changes.

---

## 1. Model family digests (named asks + HF-popular)

### 1.1 Mistral / Ministral 3

| Item | Detail |
|------|--------|
| **Refs** | Mistral 3 launch (Dec 2025); Ministral 3 [arXiv:2601.08584](https://arxiv.org/abs/2601.08584); Mistral 7B [arXiv:2310.06825](https://arxiv.org/abs/2310.06825); Small 3.1 blog |
| **Shapes** | Dense **3B / 8B / 14B** edge (Ministral) + Large MoE (≈41B active / 675B total class) |
| **Arch stack** | Decoder Transformer, **GQA** (e.g. 32Q/8KV), **RoPE**, **SwiGLU**, **RMSNorm**, large vocab (~131K), long ctx (to 128K–256K at their scale) |
| **SWA history** | Classic Mistral 7B used **sliding window attention** (window ~4K) + rolling KV buffer. **Mistral Small 3.1 appears to drop SWA** (`sliding_window: null`) and use full GQA; tradeoff speculation: SWA saves KV mem but not always latency vs FlashAttention on full attn |
| **Training** | **Cascade distillation**: iterative prune → distill → repeat from larger Mistral Small base |
| **Variants** | Base / instruct / reasoning; multimodal ViT frozen on some SKUs |

**LL3M takeaways**

- Same dense “modern stack” you already use (RMSNorm, SwiGLU, GQA/MQA, RoPE).
- **Cascade prune–distill** is the main transfer: after v75 + QAT, **teacher → 79M student** KD can beat pure CE.
- Edge positioning (Ministral) validates **small dense + recipe** over tiny MoE for sub-100M L3.
- **Do not add SWA** at T=256: adds mask complexity, zero residency win with MQA already.

---

### 1.2 NVIDIA Nemotron 3 (Nano / Super / Ultra)

| Item | Detail |
|------|--------|
| **Refs** | [Nemotron 3 Nano HF blog](https://huggingface.co/blog/nvidia/nemotron-3-nano-efficient-open-intelligent-models); Super [arXiv:2604.12374](https://arxiv.org/abs/2604.12374); Ultra tech report (NVIDIA research PDF, ~Jun 2026); Elastic [arXiv:2511.16664](https://arxiv.org/abs/2511.16664) |
| **Family** | **Nano** ~30B total / **3B active**; **Super** ~120B / **12B active**; **Ultra** ~550B / **55B active** (MoE hybrid) |
| **Arch** | **Hybrid Mamba-2 + sparse attention (GQA) + MoE FFN**. Nano: ~52 layers, macro blocks of Mamba-2 → MoE, few GQA layers; Super: LatentMoE, 88 layers class; 1M claimed context on large SKUs |
| **Why hybrid** | Mamba-2: O(T) sequence state, long-ctx throughput; sparse full attn: precision reasoning; MoE: capacity without full dense FFN cost |
| **Nano 4B path** | **Prune + distill** from larger Nano via **Elastic** (router-guided structured prune + KD); multi-stage recovery |
| **Training** | Huge token budgets; SFT; **RLVR / RLHF**; **PivotRL** (assistant-turn RL reusing offline SFT trajectories); open agentic datasets on HF |
| **Quant** | **NVFP4** pretrain efficiency; FP8 deploy SKUs |
| **Product UX** | Reasoning ON/OFF, thinking budgets (agent economics) |

**LL3M takeaways**

- **Elastic-style compress**: structured shrink + KD > inventing MoE at 79M.
- **Open post-train data** (nvidia collections): seed tool/reasoning SFT if filtered to T=256 + your tag protocol.
- Hybrid Mamba is for **long-context efficiency**; with T=256 and MQA, KV is already tiny → **defer hybrid**.
- Reasoning budget / dual mode = product UX idea (short vs long answers), not architecture.

---

### 1.3 Kimi K2 / K2.5 / “K3” status (Moonshot)

| Item | Detail |
|------|--------|
| **Refs** | Kimi K2 [arXiv:2507.20534](https://arxiv.org/abs/2507.20534); K2.5 Visual Agentic [arXiv:2602.02276](https://arxiv.org/abs/2602.02276); Kimi Linear [arXiv:2510.26692](https://arxiv.org/abs/2510.26692); HF `moonshotai/Kimi-K2*` |
| **K2 scale** | **~1.04T total MoE**, **~32B active**, 61 layers, **384 experts**, top-8 + 1 shared, SwiGLU, vocab 160K, ctx 128K→256K (K2.5) |
| **Attention** | **MLA** (DeepSeek-style multi-head latent attention); fewer heads than DeepSeek-V3 (64 vs 128) for inference |
| **Train** | **MuonClip** = Muon optimizer + **QK-Clip** (rescales Q/K weights when attention logits explode); 15.5T tokens; **zero loss spikes** claimed; synthetic data for token efficiency |
| **Post-train** | Industrial **agentic data synthesis** + joint RL (verifiable rewards + self-critic) in real/synthetic environments |
| **Tools** | Explicit multi-part template: tool declare / invoke / result |
| **K2.5** | Multimodal joint text-vision pretrain/SFT/RL; **Agent Swarm** (parallel sub-agents, up to ~4.5× latency cut) |
| **Kimi Linear** | Hybrid **KDA linear attn + MLA**; less KV (up to ~75%), competitive quality; 1M context class experiments |
| **“Kimi K3”** | Not a stable public flagship name as of mid-2026; K2.5/K2.6/Agent Swarm are the live line. Treat “K3” as user shorthand for Moonshot’s next agentic SKU |

**LL3M takeaways**

- Architecture (1T MoE, MLA) is **not** for L3.
- **Agentic data factory + verifiers** is the gold standard for tool skill.
- **QK-Clip / softcap**: when QAT or deep thin nets spike logits, clip attention logits / QK scale (you have softcap knobs).
- **Agent Swarm** idea at tiny scale: host can parallelize independent tool calls (calc + lookup) without model changes.
- Muon is optional; AdamW is fine at 79M unless you chase token efficiency on large runs.

---

### 1.4 DeepSeek V3 → V3.2 → V4 direction

| Item | Detail |
|------|--------|
| **V3** | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437): **MLA**, DeepSeekMoE, **MTP** (multi-token prediction), FP8 train |
| **V3.2** | **DeepSeek Sparse Attention (DSA)** [arXiv:2512.02556](https://arxiv.org/abs/2512.02556); FlashMLA sparse kernels; dual K-cache (indexer + MLA) |
| **mHC** | [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) *Manifold-Constrained Hyper-Connections*: multi-stream residual that restores identity mapping (stability at extreme residual width) |
| **Engram** | [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) *Conditional Memory via Scalable Lookup*: O(1) **N-gram / hash memory** as second sparsity axis beside MoE; U-shaped allocation law compute vs static memory; host-memory prefetch; gains on knowledge *and* reasoning/code; frees early layers from memorizing static patterns |
| **V4 (public trajectory, mid-2026)** | Hybrid sparse attention (**CSA/HCA** class), mHC, Engram conditional memory; dual sparsity: MoE (dynamic compute) + Engram (static lookup); long-ctx serving co-design (prefetch, disagg) |

**LL3M takeaways (highest conceptual value of any frontier paper this pass)**

| DeepSeek idea | LL3M mapping |
|---------------|--------------|
| **Engram** | Your product thesis in formal clothes: **knowledge outside the residual stream**. Host `lookup`/`wiki`/notes ≈ conditional memory. Optional: host-side hashed fact tables O(1), not in ternary weights |
| **MLA / DSA** | Long-ctx KV/compute; **low priority at T=256 MQA** |
| **MTP** | You already have shared-head MTP aux → keep/tune weight |
| **mHC** | Deep residual stability at huge scale; LayerScale + residual design already covers 79M |
| **FP8 train** | GPU throughput only |

**Critical Engram quote (paraphrased thesis):** Transformers lack a native knowledge-lookup primitive and waste compute simulating retrieval; conditional memory is a first-class sparsity axis. For LL3M: **do not pack encyclopedia into 21 MiB ternary**; use tools + host memory.

---

### 1.5 GLM-5 / GLM-5.2 (Zhipu / Z.ai)

| Item | Detail |
|------|--------|
| **Refs** | GLM-5 [arXiv:2602.15763](https://arxiv.org/abs/2602.15763); GLM-5.2 blog; IndexShare [arXiv:2603.12201](https://arxiv.org/abs/2603.12201) |
| **Scale** | ~**744B total / ~40B active**, 256 experts, ~80 layers, DSA + MLA lineage |
| **Train** | Huge token budget (~28.5T class for GLM-5); agentic engineering focus |
| **GLM-5.2** | **IndexShare**: share DSA top-k **indexer every 4 layers** → ~2.9× less indexer FLOPs at 1M ctx; improved **MTP** for speculative decode (+acceptance); 1M context product claim |
| **Local note** | Sibling `../GLM5.2` toolsmith: **retrieval-lift** for sub-1B (tools ≫ weight knowledge) |

**LL3M takeaways**

- Sparse/index attention = **long context infra**, not 256-token story/tool loop.
- **Agentic post-train + tool evaluation** aligns with agent mode.
- Double-down on **retrieval/tool skill** over stuffing facts (same as GLM5.2 FINDINGS).
- IndexShare is a pattern (“reuse expensive selection across layers”) only if you ever do sparse attention.

---

### 1.6 Gemma 4

| Item | Detail |
|------|--------|
| **Refs** | Gemma 4 Technical Report [arXiv:2607.02770](https://arxiv.org/abs/2607.02770); Google model cards |
| **Family** | Dense small (2B/4B-class) + larger dense/MoE (e.g. **26B-A4B**); natively multimodal (text/image/video/audio depending on SKU); long context (128K–256K class) |
| **Common pieces** | Modern Transformer stack; efficiency variants; open weights culture; careful norms / **logit softcap** lineage from earlier Gemma |

**LL3M takeaways**

- Small Gemma SKUs reinforce **dense SLMs for edge**.
- Softcap / careful norms → you already have softcap option; useful under QAT.
- Multimodal: out of scope for L3 text product.

---

### 1.7 DiffusionGemma / diffusion LMs

| Item | Detail |
|------|--------|
| **Refs** | Google DiffusionGemma (Jun 2026 open weights on Gemma 4 26B-A4B); LLaDA [arXiv:2502.09992](https://arxiv.org/abs/2502.09992); BD3-LM [arXiv:2503.09573](https://arxiv.org/abs/2503.09573); transparency [arXiv:2606.20560](https://arxiv.org/abs/2606.20560); constrained decode for diffusion LLMs [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) |
| **Arch** | Masked **discrete diffusion** MoE (~25.2B total / ~3.8B active); denoise **blocks** (e.g. 256 tokens) in parallel over multiple steps |
| **Claimed win** | Up to ~4× faster on high arithmetic-intensity GPUs; lower quality / more factual errors vs AR twin in community tests |
| **Caveat** | Speedup weaker on bandwidth-bound / unified-memory (Apple Silicon note in Google blog) |

**LL3M takeaways**

- **Different runtime** (many forwards over a mask canvas) vs single-token C decode.
- Constrained tool decoding is harder under diffusion.
- **Do not port** into L3T2 C path; watch as research only.
- Quality vs latency trade is real: Google positions AR Gemma for fact-sensitive work.

---

### 1.8 Qwen3 / Qwen3.5 (HF gravity well)

| Item | Detail |
|------|--------|
| **Refs** | Qwen3 [arXiv:2505.09388](https://arxiv.org/abs/2505.09388); Qwen3.5 blog (2026); Gated DeltaNet [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| **Qwen3** | Dense 0.6B–32B + large MoE; **thinking / non-thinking** unified modes; thinking budget; 119→ languages; Apache 2.0 |
| **Post-train** | 4-stage: CoT cold start → reasoning RL → mode fusion → general RL |
| **Qwen3-Next / 3.5** | **Hybrid**: ~3:1 **Gated DeltaNet (linear)** : full gated attention; sparse MoE (e.g. 397B-A17B); long-ctx efficiency |
| **Small SKUs** | Qwen3.5-0.8B etc.: hybrid layout documented on HF (GDN blocks + sparse full attn) |

**LL3M takeaways**

- **Dual-mode generation** as product UX (fast answer vs short “think” trace) without two models.
- **Data volume + distillation** from flagship into small dense is how Qwen makes 0.6B competitive.
- Hybrid GDN: only if T grows; do not rewrite C for linear attn at T=256.
- Strong **tool templates** in instruct data: study format discipline for your `<call>` tags.

---

### 1.9 SmolLM2 / SmolLM3 (Hugging Face TB)

| Item | Detail |
|------|--------|
| **Refs** | HF blogs + model cards; fully open data mixture / training blueprint |
| **Sizes** | e.g. 135M, 1.7B, **3B** (SmolLM3); multi-trillion token budgets at 3B |
| **Recipe** | **Three-stage curriculum**: web → code → math/reasoning; optional reasoning mode |
| **Arch notes** | Dense modern stack; SmolLM3 experiments include **periodic NoPE** layers (every Nth layer omits RoPE) in some gallery descriptions |
| **Why famous** | Transparency: weights + data + recipe |

**LL3M takeaways**

- **Curriculum stages** map cleanly: story/pretrain → short Q&A → tool SFT → (optional) light reasoning.
- Fully open mixes are a template for documenting your own data recipe.
- 135M class shows **tiny dense models** are a researched product niche (you sit under that).

---

### 1.10 Phi-4 / mini (Microsoft)

| Item | Detail |
|------|--------|
| **Refs** | Phi-4 model cards / technical reports; community SLM roundups 2026 |
| **Shapes** | Phi-4 ~14B; **Phi-4-mini ~3.8B**; reasoning variants |
| **Recipe** | High-quality + **synthetic textbook** data; reasoning-dense content; strong instruction following for size |
| **Edge** | Runs on CPU/NPU class devices in product narratives |

**LL3M takeaways**

- **Data quality > scale** is the Phi religion; apply to 79M (synthetic tool + short Q&A gold).
- Not an architecture fork: dense GQA Transformer.
- Full write-up of knowledge-dense small models (Phi/SmolLM/Qwen/Gemma, datasets, Engram, capacity math): **`docs/KNOWLEDGE_DENSITY_SMALL_MODELS.md`**.

---

### 1.11 Llama 3.x / Llama 4 Scout–Maverick (Meta)

| Item | Detail |
|------|--------|
| **Refs** | Meta Llama 4 blog (Apr 2025); vLLM Llama 4 notes |
| **Llama 4** | First Llama **MoE**; **Scout** ~109B total / 17B active / **16 experts**; **Maverick** ~400B / 17B active / **128 experts**; natively multimodal; huge claimed context (Scout up to 10M class in product materials) |
| **Arch extras** | Shared + routed experts; **iRoPE** (interleave global NoPE attention with local RoPE attention, ~1:3) |
| **Ecosystem** | Day-0 vLLM, community license gravity |

**LL3M takeaways**

- Ecosystem tooling, not architecture for L3.
- iRoPE/NoPE cadence is interesting only if long-ctx returns; skip for T=256.
- MoE at 79M: **reject** (router + expert residency thrash).

---

### 1.12 BitNet / ternary family (native quant)

| Item | Detail |
|------|--------|
| **BitNet** | [arXiv:2310.11453](https://arxiv.org/abs/2310.11453) 1-bit Transformers |
| **b1.58** | [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) ternary {-1,0,1} |
| **2B4T** | [arXiv:2504.12285](https://arxiv.org/abs/2504.12285) native W1.58 at 2B |
| **bitnet.cpp** | [arXiv:2502.11880](https://arxiv.org/abs/2502.11880) **TL** (ternary LUT GEMM) + **I2_S** (lossless train-matched edge) |
| **a4.8** | [arXiv:2411.04965](https://arxiv.org/abs/2411.04965) 4-bit activations + 1.58 weights |
| **TernaryLLM / Tequila / CAT-Q / TernaryLM** | Recipes for QAT vs PTQ; absmean vs learnable Δ/α; small-net native ternary (e.g. TernaryLM ~132M [arXiv:2602.07374](https://arxiv.org/abs/2602.07374)) |
| **T-SAR** | [arXiv:2511.13676](https://arxiv.org/abs/2511.13676) CPU SIMD ternary co-design; bandwidth critique of naive TLUT |

**LL3M takeaways (your quant core)**

- **Absmean + STE** is the open-source default for stable ternary QAT.
- **Train–serve grid lock** (I2_S lesson): same scales/pack as C L3T2.
- **QAT >> post-hoc** for quality; PTQ ternary still weak vs long QAT (CAT-Q tries to close PTQ gap).
- Small nets need **width/care** (BitNet small-net paper): your d576×L23 is intentional.
- Optional later: W1.58A8, TL-style GEMV if tok/s bound.

---

### 1.13 Other HF-popular / edge lines (short)

| Family | Why it matters |
|--------|----------------|
| **Granite 4.x** (IBM) | Enterprise open weights; tool/RAG recipes |
| **Aya / multilingual SLMs** | Multilingual data discipline (if you ever leave EN) |
| **MobileLLM / Gemma 3n** | Width–depth for phones; residency mindset |
| **OLMo 2** | Fully open training stack (Allen AI); recipe culture |
| **GPT-OSS / community MoE** | Architecture comparison fodder only |
| **TriLM / Spectra / BitCPM** | Community ternary; prefer native QAT over PTQ ports |

---

## 2. Deep paper themes by problem

### 2.1 Ternary / 1.58-bit / edge kernels

| Paper / system | ID | Transfer |
|----------------|-----|----------|
| BitNet | 2310.11453 | BitLinear, STE |
| BitNet b1.58 | 2402.17764 | Ternary scaling narrative |
| BitNet 2B4T | 2504.12285 | Native W1.58A8 at 2B |
| bitnet.cpp | 2502.11880 | TL + I2_S lossless edge |
| BitNet a4.8 | 2411.04965 | Act quant |
| TernaryLLM | 2406.07177 | Ternarization recipes |
| T-SAR CPU | 2511.13676 | SIMD; TLUT bandwidth |
| BitNet small nets | 2407.09527 | Small-net care |
| Tequila (trapping-free ternary) | 2509.23809 | Absmean QAT stability; retrain recovery |
| CAT-Q | 2606.26650 | PTQ ternary: learnable α/Δ + softened ternarization (still secondary to QAT) |
| TernaryLM | 2602.07374 | Native ternary ~132M; adaptive layer scales; STE |

**Already on quant backlog:** QAT, train–serve lock, BitLinear, optional A8, optional TL kernels.

### 2.2 Architecture efficiency (KV, sparse, hybrid, memory)

| Paper / idea | ID | Transfer |
|--------------|-----|----------|
| DeepSeek-V3 MLA/MoE/MTP | 2412.19437 | MTP aux; MLA if T grows |
| DeepSeek Sparse Attention | 2512.02556 | Long-ctx; skip at T=256 |
| mHC hyper-connections | 2512.24880 | Extreme residual stability; skip |
| **Engram conditional memory** | **2601.07372** | **External O(1) knowledge; host tools/tables** |
| IndexShare | 2603.12201 | Share sparse indexers; skip |
| Kimi K2 MLA+MoE | 2507.20534 | Agentic post-train |
| Kimi Linear | 2510.26692 | Hybrid linear+MLA; long ctx |
| Gated DeltaNet | 2412.06464 | Linear attn component in Qwen3.5 |
| Ministral 3 | 2601.08584 | Dense edge + cascade distill |
| Nemotron Elastic | 2511.16664 | Joint prune + KD |
| Nemotron 3 Super | 2604.12374 | Hybrid Mamba-MoE-Attention at scale |

### 2.3 Small models, distill, data

| Paper / idea | ID / source | Transfer |
|--------------|-------------|----------|
| SLMs still pack a punch | 2501.05465 | Survey of SLM design + data (~160 papers) |
| SmolLM2/3 | HF TB | Curriculum mixes; open data |
| Phi series | MS cards | Synthetic + textbook quality |
| Cascade distill (Ministral) | 2601.08584 | Prune–distill loop |
| Qwen3 report | 2505.09388 | Think modes; distill small from large |

### 2.4 Agents / tools / constrained decode

| Paper / idea | ID / source | Transfer |
|--------------|-------------|----------|
| Kimi K2 agentic synthesis + RL | 2507.20534 | Verified tool trajectories |
| Kimi K2.5 Agent Swarm | 2602.02276 | Parallel tool orchestration (host-side) |
| Nemotron agentic data | HF nvidia | Filterable SFT fuel |
| GLM-5 agentic engineering | 2602.15763 | Long-horizon tools |
| SLMs for agentic tool calling | 2512.15943 | Targeted FT can beat larger baselines on tool tasks |
| Small models function calling study | 2504.19277 | Format failure modes; need constrained decode |
| In-vehicle SLM function calling | 2501.02342 | Special tokens for tools (Octopus-style) |
| SimpleTool parallel FC | 2603.00030 | Parallel decode of name/args; latency |
| Guided structured templates | 2509.18076 | Template curriculum for FC |
| BFCL / When2Call / xLAM | datasets | When-to-call balance |
| Local GLM5.2 FINDINGS | sibling project | Retrieval-lift metric |

### 2.5 Optimizers / train stability

| Paper / idea | ID | Transfer |
|--------------|-----|----------|
| Muon scalable | 2502.16982 | ~2× token efficiency vs AdamW at large scale; weight decay + update scale |
| MuonClip + QK-Clip | via Kimi 2507.20534 | Prevent attention logit explosion |
| Softcap (Gemma lineage) | model cards | You have this; use under QAT |

### 2.6 Diffusion LMs (research only)

| Paper / system | ID | Transfer |
|----------------|-----|----------|
| LLaDA | 2502.09992 | AR-free diffusion LM |
| BD3-LM | 2503.09573 | Block diffusion |
| DiffusionGemma | Google 2026 | Fast block denoise; quality tradeoff |
| Constrained decode diffusion LLMs | 2508.10111 | Tools hard under diffusion |

---

## 3. Cross-cutting design patterns → LL3M

### 3.1 Already adopted (keep)

| Pattern | Frontier use | LL3M |
|---------|--------------|------|
| RMSNorm + pre-norm | Universal | Yes |
| SwiGLU FFN | Universal | Yes |
| GQA/MQA | Universal | **MQA product** |
| RoPE (rotate-half aligned train/serve) | Universal | Yes (goldens 24/24) |
| QK-Norm | Qwen / modern | Yes |
| LayerScale | Deep thin nets | Yes (v75) |
| Ternary W pack L3T2 | BitNet | **Yes** |
| Host tool loop | All agent models | **agent mode** |
| MTP aux | DeepSeek | Optional flag |
| Softcap | Gemma | Optional flag |

### 3.2 High value, not yet (or partial)

| Pattern | Why help | Effort |
|---------|----------|--------|
| **Native ternary QAT** matching C | BitNet quality story | **Queued / in flight** |
| **BitLinear** first-class | Clean train/export | Medium |
| **Synthetic agentic data + verifiers** | Kimi / Nemotron / GLM | Medium (data) |
| **Teacher distill into 79M** | Ministral / Elastic / Phi / Qwen | Medium–high |
| **Data curriculum stages** | SmolLM / Qwen | Medium |
| **Constrained tool decode** | SLM tool papers | Medium |
| **I2_S / train-matched kernels** | bitnet.cpp | Medium (C) |
| **Host Engram-lite** (hash fact table) | Engram 2601.07372 | Medium (host, not weights) |
| **W1.58A8** | BitNet a4.8 | Higher risk at 79M |
| Hybrid Mamba / GDN / MLA / DSA | Long-ctx frontier | **Low value at T=256** |
| Diffusion LM | Latency experiments | **Out of product path** |
| Muon optimizer | Token efficiency | Optional train experiment |

### 3.3 Anti-patterns for L3 product

| Temptation | Why reject |
|------------|------------|
| Port MoE at 79M | Router + expert memory thrash; gate is **bytes** |
| Chase 128K–1M context in weights | Blows KV and residency claim |
| Store world knowledge in ternary weights | Density math fails; Engram/tools win |
| Free-form JSON tools without SFT | SLMs fail format |
| Copy DiffusionGemma into C decode | Different compute graph |
| SWA “because Mistral did” | Dropped in newer Small; useless at T=256 |
| NVFP4 / GPU-only pretrain as serve path | Serve is ternary CCD on CPU |

---

## 4. Architecture comparison table (what each lab optimizes)

| Family | Capacity axis | Sequence axis | Memory/knowledge axis | Edge relevance |
|--------|---------------|---------------|------------------------|----------------|
| Mistral/Ministral | Dense (edge) / MoE (large) | Full GQA (SWA legacy) | Parametric | High (dense edge) |
| Nemotron 3 | MoE FFN | Mamba-2 + sparse attn | Parametric + agent data | Medium (recipes) |
| Kimi K2 | Ultra-sparse MoE | MLA (+ Linear research) | Agentic synthesis | High (tools) |
| DeepSeek V3–V4 | MoE | MLA → DSA → CSA/HCA | **Engram** + MoE | High (thesis) |
| GLM-5.2 | MoE | DSA + IndexShare | Agentic | High (tools) |
| Qwen3.5 | Sparse MoE | GDN hybrid | Parametric + tools | Medium (small SKUs, UX modes) |
| Gemma 4 | Dense/MoE | Full modern | Multimodal parametric | Medium (softcap, dense SLM) |
| DiffusionGemma | MoE + diffusion head | Block denoise | Parametric | Low (runtime mismatch) |
| BitNet | Dense ternary | Full | Compress weights | **Highest (kernels)** |
| **LL3M (us)** | Dense 79M | MQA T=256 | **Host tools** | **Product** |

---

## 5. Concrete next experiments

These extend `AGENTS.md` quant backlog with frontier research items:

| # | Experiment | Success metric | Source inspiration |
|---|------------|----------------|--------------------|
| A | Finish QAT compare (automated) | `quality_compare.json` serve-grid ΔCE | BitNet |
| B | **Gold tool-SFT mix** (calc/lookup/time + no-call ~1:1) then SFT | Agent mode works with `--no-assist-tools` | Kimi / BFCL / When2Call |
| C | **KD from Qwen3-0.8B/1.7B or Ministral-3B** into v75 shape | Val CE + sample quality vs CE-only | Ministral cascade, Elastic |
| D | Stage data mix: story → short Q&A → tools | Capability report + tool accuracy | SmolLM curriculum |
| E | Constrained decode for `<call>…</call>` | Format exactness on held-out queries | SLM FC papers, SimpleTool |
| F | Port **TL-style** ternary GEMV if tok/s < target | Median tok/s, residency still PASS | bitnet.cpp, T-SAR |
| G | (Optional) int8 activation QAT + C | ΔCE under A8; WS impact | BitNet a4.8 |
| H | Host **Engram-lite**: hashed offline fact KV for `lookup` | Hit rate + latency; no weight growth | Engram 2601.07372 |
| I | Parallel host tool fan-out when model emits multi-call | Wall latency on multi-tool queries | Kimi Agent Swarm (host) |
| J | Softcap / QK clamp ablation during QAT | Loss spike rate, final CE | Gemma, Kimi QK-Clip |

---

## 6. Bibliography (primary)

### Quant / ternary / edge
1. Wang et al., *BitNet*, [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)  
2. Ma et al., *BitNet b1.58*, [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)  
3. *BitNet b1.58 2B4T*, [arXiv:2504.12285](https://arxiv.org/abs/2504.12285)  
4. Wang et al., *bitnet.cpp*, [arXiv:2502.11880](https://arxiv.org/abs/2502.11880)  
5. *BitNet a4.8*, [arXiv:2411.04965](https://arxiv.org/abs/2411.04965)  
6. *TernaryLLM*, [arXiv:2406.07177](https://arxiv.org/abs/2406.07177)  
7. *T-SAR* CPU ternary, [arXiv:2511.13676](https://arxiv.org/abs/2511.13676)  
8. Nielsen et al., BitNet small nets, [arXiv:2407.09527](https://arxiv.org/abs/2407.09527)  
9. Tequila, [arXiv:2509.23809](https://arxiv.org/abs/2509.23809)  
10. CAT-Q, [arXiv:2606.26650](https://arxiv.org/abs/2606.26650)  
11. TernaryLM, [arXiv:2602.07374](https://arxiv.org/abs/2602.07374)  

### Frontier open models / architecture
12. DeepSeek-V3, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)  
13. DeepSeek Sparse Attention, [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)  
14. mHC, [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)  
15. **Engram**, [arXiv:2601.07372](https://arxiv.org/abs/2601.07372)  
16. Kimi K2, [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)  
17. Kimi K2.5, [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)  
18. Kimi Linear, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)  
19. GLM-5, [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)  
20. IndexShare, [arXiv:2603.12201](https://arxiv.org/abs/2603.12201)  
21. Ministral 3, [arXiv:2601.08584](https://arxiv.org/abs/2601.08584)  
22. Mistral 7B, [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)  
23. Gemma 4 Technical Report, [arXiv:2607.02770](https://arxiv.org/abs/2607.02770)  
24. Nemotron Elastic, [arXiv:2511.16664](https://arxiv.org/abs/2511.16664)  
25. Nemotron 3 Super, [arXiv:2604.12374](https://arxiv.org/abs/2604.12374)  
26. Qwen3, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)  
27. Gated DeltaNet, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)  
28. Muon scalable, [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)  

### Small models / distill / tools
29. *SLMs Can Still Pack a Punch*, [arXiv:2501.05465](https://arxiv.org/abs/2501.05465)  
30. SmolLM2/3 (HF blogs + datasets)  
31. Phi-4 model cards (Microsoft)  
32. SLMs agentic tool calling, [arXiv:2512.15943](https://arxiv.org/abs/2512.15943)  
33. SLM function calling study, [arXiv:2504.19277](https://arxiv.org/abs/2504.19277)  
34. In-vehicle SLM FC, [arXiv:2501.02342](https://arxiv.org/abs/2501.02342)  
35. SimpleTool, [arXiv:2603.00030](https://arxiv.org/abs/2603.00030)  
36. Guided structured FC templates, [arXiv:2509.18076](https://arxiv.org/abs/2509.18076)  

### Diffusion LMs
37. LLaDA, [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)  
38. BD3-LM, [arXiv:2503.09573](https://arxiv.org/abs/2503.09573)  
39. DiffusionGemma transparency, [arXiv:2606.20560](https://arxiv.org/abs/2606.20560)  
40. Constrained decoding diffusion LLMs, [arXiv:2508.10111](https://arxiv.org/abs/2508.10111)  

### Systems / local
41. Microsoft BitNet / bitnet.cpp GitHub  
42. NVIDIA Nemotron open data + recipes (HF `nvidia`)  
43. Sebastian Raschka LLM Architecture Gallery / 2025 comparison series  
44. Local sibling: `../cwen` (CPU ternary GEMV), `../GLM5.2` (tool/retrieval-lift)

---

## 7. Decision table for *this* repo

| Decision | Recommendation | Why (research-backed) |
|----------|----------------|----------------------|
| Change v2 block to MoE/MLA/Mamba now? | **No** | Capacity/sequence wins target huge T and GPU; L3 gate is bytes at T=256 |
| Proceed ternary QAT? | **Yes** — prefer **native from step 0** | BitNet, Tequila, TernaryLM; see `docs/NATIVE_TERNARY_PIPELINE.md` |
| Invest in agentic synthetic data? | **Yes** (highest non-quant ROI) | Kimi, Nemotron, GLM, SLM tool papers |
| Distill from 0.8B–3B teacher? | **Yes after QAT baseline** | Ministral cascade, Elastic, Qwen small |
| Port DiffusionGemma? | **No** for product C path | Wrong compute graph; quality/facts trade |
| Grow T to 4k+? | Only with residency re-budget | Breaks ~30 MiB story |
| Treat BitNet as architecture fork? | **No** | Quant + kernel + train alignment only |
| Engram inside ternary weights? | **No** | Host/tools *are* the Engram for L3 |
| Host Engram-lite fact tables? | **Optional P2** | Formalizes lookup; free weight budget |
| “Kimi K3” chase? | **N/A** | No stable public K3; track K2.5 agent stack |

---

## 8. One-page “what we learned that changes nothing / changes something”

### Confirms product freeze (changes nothing)
- Dense modern stack (RMSNorm, SwiGLU, RoPE, MQA) is still the right skeleton for sub-100M edge.
- Knowledge density problem is universal; frontier is *also* externalizing knowledge (Engram, tools, RAG).
- MoE/hybrid/sparse attention are primarily **server-scale sequence and capacity** tools.

### Changes or strengthens near-term plan
1. **QAT + train–serve lock** remains P0 (BitNet ecosystem consensus).  
2. **Tool SFT + verifiers** is as important as architecture papers (every agent lab).  
3. **KD from small open teachers** is the proven path for edge quality (Ministral, Elastic, Qwen).  
4. **Curriculum data stages** beat “more of the same mix” (SmolLM, Phi).  
5. **Engram** legitimizes host-side knowledge as first-class research, not a hack.  
6. **Constrained decode** is required for reliable SLM tools, not optional polish.  
7. **Softcap / QK clamp** under QAT is cheap insurance (Gemma, Kimi).  

---

## 9. Colibrì runtime learnings ([JustVugg/colibri](https://github.com/JustVugg/colibri))

Pure-C MoE serve for **GLM-5.2 (744B)** on ~25 GB RAM (experts streamed from disk).
Not a ternary SLM recipe — but high-signal **serve/policy** lessons for LL3M.

| Colibrì idea | Transfer to LL3M | Status |
|--------------|------------------|--------|
| Never silently change precision/router to “fit” memory | L3 gate: refuse or `--force`; no silent de-quant | already product rule |
| Pure C + transformers oracle goldens | Keep `compare_c_golden` / RoPE goldens for any new kernel | keep |
| MTP draft head needs **more bits** than bulk (int8 vs int4) | Train MTP under STE OK; if serve-time draft ever added, keep draft head higher precision | document |
| Grammar-forced drafts for tools/JSON | Constrained `<call>…</call>` post-process + clip babble | **`ll3m/constrained.py` + agent** |
| Calmer sampling under quant (temp 0.7 / tight nucleus) | Defaults temp **0.7** story, **0.4**/top_k **20** agent | **chat/generate defaults** |
| Measure int kernels **per shape** (int4 1×row sometimes slower than f32) | TL ternary GEMV: A/B decode before default | backlog bn7 |
| Mixed precision by sensitivity (norms f32) | Norms/scales f32, embeds int8, Linears ternary | already |
| Disk/VRAM tier hierarchy | Only if WS ever leaves full L3 residency | skip for freeze |

**Do not copy:** MoE disk expert stream as a substitute for the CCD product claim.

---

## 10. Maintenance

- Refresh when a major open release lands (new Mini*/Nano/Phi/Qwen edge SKU, BitNet kernel paper, or a real Kimi K3 tech report).  
- Keep **P0/P1** mirrored in `AGENTS.md` quant backlog.  
- Sibling research: `../cwen` (CPU ternary GEMV craft), `../GLM5.2` (tool/retrieval-lift for tiny models).  
- Colibrì: re-check if they ship denser pure-C kernels useful for ternary GEMV.  
- Cite primary arXiv IDs in new experiment notes so claims stay falsifiable.
