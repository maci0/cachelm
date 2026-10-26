# LL3M deep overview

Companion to the [README](../README.md). Commands and file layout live there;
this page is the conceptual map.

## 1. Problem statement

Most LM work optimizes quality alone. **LL3M** asks:

> How much next-token capability can we get if the **entire inference data
> working set** must fit in **one CPU L3 CCD**, stay there (no DRAM on the hot
> path after warm-up), and we refuse to hand-wave Python process size?

That forces:

- A real but small architecture  
- Aggressive serve quant (int8 or **ternary**)  
- A C runtime with contiguous, mlock’d arenas  
- Metrics that match the corpus (not MMLU cosplay)  

Training may use all of DRAM/VRAM. Only **C serve** carries the claim.

## 2. Data

### TinyStoriesV2-GPT4 (primary for v2)

Narrative English written so **small** models can learn fluent story language
([TinyStories paper](https://arxiv.org/abs/2305.07759)). Prefer the
**GPT-4-only** V2 release (higher quality than mixed GPT-3.5/4). Default download
is a **~1.5 GiB** train partial via `scripts/download_data.sh` (`MAX_BYTES=0` for
the full ~2.2 GiB file) plus the official valid set.

### Tiny Shakespeare (style + legacy v1)

`data/tinyshakespeare.txt` (~1.1 MB): public-domain plays, classic char-LM demo
corpus (Karpathy / nanoGPT lineage). Used as a small style spice in the mix, not
as the main corpus.

### Mix (default train, tuned for ~75M)

`scripts/prepare_data.py` builds:

- Story-level split on `<|endoftext|>` / blank lines (no char-cut leakage)  
- Drops incomplete tail story on byte-range partials  
- Quality filters (printable/alpha density, word/bigram spam, sentence-ish end)  
- Near-exact dedup + train/val cross-dedup  
- Official GPT-4 valid when present (pure stories, no Shakespeare)  
- **~95%** GPT-4 TinyStories + **~5%** Shakespeare (train only), interleaved  
- Domain **BPE V=2048** → `tokenizer.json` (stratified 40M-char sample default)  
- Pre-encoded `mix_train.bin` / `mix_val.bin` + `mix_manifest.json`  

Current rebuild (see `data/mix_manifest.json`): **~466M train tokens**, ~6.0M
val tokens, ~1.95M train stories.

## 3. Tokenization

| Mode | Vocab | Use |
|------|-------|-----|
| **BPE** (default) | 2048 | v2; more English per context length |
| **Char** | 65 on Shakespeare | v1 legacy |

BPE is **byte-level** domain-trained (pure Python in `ll3m/tokenizer.py`).
Larger vocabs (8k–150k) waste L3 on embeddings at this scale.

## 4. Model math (sketch)

### Embeddings

```
h[t] = wte[id[t]]          # v2: + RoPE later in attention (no wpe table)
h[t] = wte[id[t]] + wpe[t] # v1 only
```

### Causal attention (v2)

GQA: fewer K/V heads than Q. RoPE on Q/K. QK-Norm. No bias. Causal SDPA.

### MLP (v2)

SwiGLU: `down(silu(gate(x)) * up(x))`, intermediate ≈ (8/3)·d.

### Loss

Mean cross-entropy next-token NLL. Generation: sample or greedy from last logits.

## 5. Train vs serve memory

| Phase | Where | Budget |
|-------|--------|--------|
| Train | DRAM / GPU VRAM | Unconstrained (VRAM fraction capped for stability) |
| Serve | C arenas | **≤ one L3 CCD**, pinned + warm |

GQA shrinks **KV**. Ternary shrinks **Linear weights**. Both help the serve gate.

## 6. Quantization summary

| Stage | Mechanism |
|-------|-----------|
| Train optional | `--qat int8` or `--qat ternary` (STE fake-quant) |
| C load | Ternary 2-bit pack (default) or int8 absmax |
| Always f32 | Embeddings, RMS/Layer norms, activations, KV |

Details: [QUANTIZATION.md](QUANTIZATION.md).

## 7. L3 residency summary

**Serve numbers (product denser-C, reference host):** hard+FIFO single-stream ~**280–282 tok/s**; multi-session under the same CCD gate up to **N=8** (~2k aggregate tok/s with shorter KV) or **N=2** at full T=256 (~550 aggregate). See [L3_RESIDENCY.md](L3_RESIDENCY.md).

### Checklist

| Mechanism | Role |
|-----------|------|
| Fit WS in 32 MiB CCD | Capacity |
| mlock + prefault | No reclaim / soft faults |
| Bind one CCD | No split across two L3 slices |
| Warm forwards | Steady-state set in L3 |
| perf DRAM share | Proof (~0–5% fills) |

Details: [L3_RESIDENCY.md](L3_RESIDENCY.md).

## 8. What “capability” means here

- Lower val NLL/BPC on the **train distribution**  
- Coherent samples in story / dialogue style  
- Domain preference on probes (char eval suite)  
- **Plus** L3 residency gates still green  

Not claimed: general chat as a 70B, parametric world knowledge, multilingual AGI.

**In product path (host-assisted):** short tool protocol (`docs/TOOL_CALLING.md`);
weights hold language + tool **policy**, not an encyclopedia (see Engram /
[RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md)).

## 9. Docs map (research + product)

| Doc | Role |
|-----|------|
| [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) | Deep model + arXiv survey → actions |
| [RESEARCH.md](RESEARCH.md) | Earlier architecture tweak ranking |
| [QUANTIZATION.md](QUANTIZATION.md) | Ternary / QAT serve pack |
| [TOOL_CALLING.md](TOOL_CALLING.md) | Agent protocol |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Live train/QAT + follow-ups |
| [../AGENTS.md](../AGENTS.md) | Ops + backlog |

## 10. Glossary

| Term | Meaning here |
|------|----------------|
| **CCD** | Core complex die; one 32 MiB L3 on 9950X |
| **Infer WS** | C weight arena + state arena (acts+KV+tokens) |
| **L3T2 / L3M2** | Export magics (ternary / f32); legacy CLT2/CLM2 |
| **BPC** | Bits per character/token (NLL / ln 2) |
| **QAT** | Quantization-aware training (fake-quant STE) |
| **Ternary** | Weights in {-1,0,1} with scale; 2-bit packed in C |
| **GQA / MQA** | Grouped- / multi-query attention (product = MQA) |
| **Engram** | External O(1) knowledge memory (DeepSeek); host tools ≈ our Engram |
