# Research map: capability tweaks for ll3m

> **Canonical deep survey (models + arXiv, 2026-07-17):**  
> **[RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md)**  
> Covers Mistral/Ministral, Nemotron 3, Kimi K2/K2.5, GLM-5/5.2, DeepSeek V3–V4
> (Engram, mHC, DSA), Gemma 4, DiffusionGemma, Qwen3/3.5, SmolLM, Phi, Llama 4,
> BitNet/ternary, tool-calling papers, bibliography, decision table, experiments A–J.

This file is the **earlier architecture-tweak map** (still useful for ranked
dense knobs). Prefer the deep pass for product decisions and paper IDs.

**Product candidate freeze:** see [PRODUCT_CANDIDATE.md](PRODUCT_CANDIDATE.md)
and [TARGET_75M.md](TARGET_75M.md) (**L23×d576 MQA**, ~79M, ternary ~30.3 MiB /
30.4 MiB gate, peri-LN + LayerScale + MTP + z-loss). Older L18×d320 notes below
are historical.

Survey of modern dense-model ideas (2024–2026) filtered for **ll3m**
constraints:

1. Infer working set ≤ one L3 CCD (~32 MiB)
2. Prefer simple portable **C** serve (no MoE routing, no exotic KV schemes unless
   they clearly shrink WS at T≈256)
3. ~10–20M param class, not 1B+

Capability ranking for this project is roughly:

> **data quality/volume ≫ train objective ≫ depth/shape ≫ exotic attention**

---

## 1. Lab research map (DeepSeek, DeepMind/Google, others)

Frontier work still sits on a **decoder Transformer**. 2025–2026 papers mostly
optimize **long context, MoE scale thrift, and train signal**, not a new
paradigm for 20M + T=256 + L3 C serve.

### DeepSeek (V2 → V3 → V4)

| Idea | What it is | For ll3m |
|------|------------|-------------|
| **MLA** (multi-head latent attention) | Compress per-token KV into a latent; huge for long decode | **Skip.** Our MQA KV at T=256 is already tiny; C rewrite not worth it |
| **DeepSeekMoE** + aux-loss-free balance | Sparse experts, fine-grained routing | **Skip.** All experts want RAM; breaks simple L3 claim |
| **MTP** (multi-token prediction) | Extra future-token losses; full DeepSeek uses small MTP **blocks** + shared head; ablations show gains | **Have shared-head MTP** (`--mtp`). Full MTP modules = more params, optional later |
| **Muon optimizer** | Second-order-ish / Newton-Schulz style updates used in large trains | **Defer.** Optional train experiment; AdamW fine at 79M |
| **mHC** (manifold-constrained hyper-connections) [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) | Widen residual into *n* streams with manifold-projected mix | **Interesting but later.** LayerScale + peri-LN is our cheap residual polish now |
| **DSA / CSA/HCA** (sparse / compressed attention) | Long-context sparse selection | **Skip.** Designed for 128k–1M, not T=256 |
| **Engram** conditional memory [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) | O(1) N-gram/hash lookup as sparsity axis; host prefetch | **Adopt as product thesis:** host tools / fact tables, **not** in ternary weights. Full write-up in [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) §1.4 |
| **FP8 / storage / systems** | Train/serve systems at scale | N/A for our ROCm toy train |

DeepSeek’s transferable lesson for us: **richer train objectives (MTP) + external
knowledge (Engram/tools) + dense stack first**, not copying MLA/MoE/CSA into a
~21–30 MiB ternary kernel.

### Google DeepMind / Google Research

| Idea | What it is | For ll3m |
|------|------------|-------------|
| **Gemma 2** softcap | `cap * tanh(logits/cap)` on attn + final | Optional final softcap (code ready); Gemma 3 **replaced softcap with QK-Norm** (we have QK-Norm) |
| **Gemma 2/3 peri-LN** | RMSNorm before and after sublayers | **Have** `--peri-ln` |
| **Gemma 3** QK-Norm + GQA + local/global 5:1 | Long-context KV thrift | QK-Norm + GQA yes; SWA **skip** at T=256 |
| **Gemma 4** cross-layer KV share | Later layers reuse earlier K/V | Marginal at T=256; complicates C; skip for now |
| **Gemma 4 PLE** (per-layer embeddings) | Extra embedding capacity without fat stack | **Possible later** for density (embed-heavy, compute-light) but grows f32 tables in L3 |
| **Gemma QAT / edge** | Serve-side quant for mobile | Aligns with our **ternary QAT phase** |
| **Titans / MIRAS / Nested Learning / Hope** | Test-time memory, multi-timescale optim, continual learning | **Research only** for us; not a drop-in dense block |

DeepMind lesson: **stability norms (peri + QK-Norm) and edge quant**; long-memory
architectures are a different product than L3-resident story LM.

### Meta / Microsoft / HF / others

| Idea | Lab | For ll3m |
|------|-----|-------------|
| LLaMA recipe (RMSNorm, RoPE, SwiGLU, GQA) | Meta | **Core v2** |
| MobileLLM deep-thin + tied embeds | Meta | **L18×d320** + tied wte |
| Phi textbook / synthetic data | Microsoft | Data quality path (GPT-4 TinyStories) |
| SmolLM2 multi-stage mix + WSD | HF | **Open** recipe improvement (curriculum, schedule) |
| OLMo post/peri-norm + z-loss | Ai2 | **Have** peri-LN + z-loss |
| OpenELM / Laguna layer-wise head budgets | Apple / Poolside | Skip until we need long-T budgeting |
| CCA (compressed conv attention) | Zyphra | Long-context; skip |
| Reasoning / RLVR (R1-style) | DeepSeek + industry | Post-base **policy** stage, not denser weights |

### Theme of 2025–2026 (Raschka surveys, etc.)

> Most new open arches are **long-context cost** tricks (KV share, MLA, CSA/HCA,
> SWA). Modeling quality still tracks **data + train recipe + residual stability**.
> The block is still GPT-decoder-shaped.

For ll3m that means: **do not chase V4 attention**. Chase **data, MTP/z-loss,
depth, peri/LayerScale, ternary QAT**.

---

## 2. Already in ll3m

| Idea | Where |
|------|--------|
| RMSNorm, RoPE, SwiGLU, GQA, QK-Norm, no bias | `arch=v2` |
| Tied embeddings | v2 default |
| Partial RoPE | `--rope-partial` |
| Final logit softcap | `--logit-softcap` |
| PaLM z-loss | `--z-loss` (default 1e-4) |
| Shared-head multi-token prediction | `--mtp` / `--mtp-depth` |
| Peri-LN (post residual RMSNorm) | `--peri-ln` |
| Residual LayerScale | `--layer-scale-init` |
| Ternary / int8 QAT | `--qat` |
| GPT-4 TinyStories mix + story split | `scripts/download_data.sh`, `prepare_data.py` |

---

## 3. Ranked next moves (capability per unit complexity)

### A. Do next (highest ROI)

| # | Tweak | Why | Cost |
|---|--------|-----|------|
| 1 | **More tokens / longer train** on GPT-4 mix | Scaling laws; we only recently expanded data | GPU time |
| 2 | **Deep-thin shapes** under same WS | MobileLLM: depth > width for sub-billion | Retune L/d |
| 3 | **Keep MTP + z-loss on** for fresh trains | Richer gradient; zero serve cost | Already wired |
| 4 | **Curriculum / multi-stage mix** | SmolLM2: stages with more “hard” text later | Data recipe |
| 5 | **Simple synthetic probes** in mix | Phi lesson: textbook-style reasoning snippets (even tiny) | Data gen |
| 6 | **Peri-LN + LayerScale** when going deeper (L≥14) | OLMo/Gemma stability for depth | Small params |

Suggested deep-thin targets (illustrative, re-check `estimate_infer_ws`):

| Shape | Params (order) | Rationale |
|-------|----------------|-----------|
| L10 d320 kv2 (current) | ~11.8M | Balanced |
| L14 d288 kv2 | similar WS | More depth |
| L16 d256 kv2 | similar WS | MobileLLM-style thin |
| L12 d384 kv2 | larger | Only if ternary frees room |

### B. Worth experiments

| Tweak | Notes |
|-------|--------|
| `--rope-partial 0.5` | LLaMA-3.1; free dims for content, mild |
| `--logit-softcap 30` | Gemma; may be redundant with QK-Norm |
| WSD LR (warmup–stable–decay) | SmolLM2; swap cosine for long runs |
| Longer `block_size` (384–512) if ternary WS allows | More narrative context |
| Label smoothing 0.05 | Sometimes helps tiny models; eval carefully |
| Distill from a larger story model | Soft labels; train-only |

### C. Skip or defer for ll3m

| Tweak | Why not (now) |
|-------|----------------|
| **MoE** | Active params help quality but **all experts** may need to live nearby; routing + C complexity; L3 claim muddies |
| **MLA** | Designed for long context / huge KV; our T=256 + GQA already tiny KV |
| **Sliding window / hybrid global** | Useful at 4k–128k context; little gain at T=256 |
| **Differential attention** | Extra QK path; mixed small-model evidence; C cost |
| **GDN / Mamba / linear attn hybrid** | Research-hot; full C rewrite; marginal at short T |
| **mHC / hyper-connections** | DeepSeek-scale stability trick; unproven ROI at 10M |
| **150k BBPE** | Embed table eats L3 |
| **Muon / SOAP optimizers** | Interesting; stick to Lion/AdamW until baseline solid |

---

## 4. Data is still the biggest architecture

Empirical theme from Phi, SmolLM2, TinyStories:

- **Quality and composition** move small models more than another attention variant.
- Already done: GPT-4 TinyStories, official val, story-level split, dedup, light Shakespeare spice.
- Still open for capability:
  1. Full 2.2 GiB GPT-4 train (`MAX_BYTES=0`) if disk allows
  2. Stage-2 mix: more dialogue / “why” questions / counting in simple English
  3. Pack stories with clear EOT so the model learns boundaries
  4. Train longer: more tokens seen (multi-epoch OK if val plateaus cleanly)

---

## 5. Train recipe (recommended default after research)

```bash
# data (once)
bash scripts/download_data.sh   # or MAX_BYTES=0
uv run python scripts/prepare_data.py --vocab-size 2048

# deep-thin-ish v2 with modern train objectives
.venv/bin/python -m ll3m.train \
  --arch v2 --tokenizer-mode bpe \
  --n-layer 14 --n-embd 288 --n-head 8 --n-kv-head 2 --block-size 256 \
  --batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.40 \
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
  --peri-ln --layer-scale-init 1e-2 \
  --dropout 0.15 --lr 3e-4 --steps 30000 \
  --device rocm --infer-scope ccd \
  --prompt "Once upon a time" \
  --out-dir checkpoints/v2_deep
```

Eval NLL without MTP/z-loss (train loop already uses pure CE in `eval_loss`).

Serve: export as usual; MTP heads are **not** extra weights (shared head). Peri-LN and
LayerScale **are** weights and must be present in C CLM2 when that path lands.

---

## 6. Mental model

```
capability
   ▲
   │     ┌── exotic attn / MoE  (diminishing for L3-small)
   │    ┌┴─ depth shape / norms / scales
   │   ┌┴── train objective (MTP, z-loss, schedule)
   │  ┌┴─── data volume + quality + curriculum
   └──┴──────────────────────────────────────► complexity / risk to C+L3
```

For ll3m, climb the bottom of that stack first. Architecture has largely
**converged**; the remaining free lunch is **better data + longer training +
depth-aware regularization**, not reinventing attention.

---

## 7. References (entry points)

- TinyStories: arXiv:2305.07759  
- MobileLLM: arXiv:2402.14905  
- DeepSeek-V3 (MLA, MTP): arXiv:2412.19437  
- Gemma 2 (softcap, peri norms): arXiv:2408.00118  
- OLMo 2 (norm placement, QK-Norm): model card / technical report 2025  
- Peri-LN analysis: arXiv:2502.02732  
- MTP curriculum for SLMs: arXiv:2505.22757  
- Raschka “Big LLM Architecture Comparison” / “Recent developments in LLM architectures” (2025–2026 surveys)  
- SmolLM2 blog / tech notes (multi-stage data)  
- Phi series tech reports (data quality)  
