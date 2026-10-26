# Improvements changelog

Capability and residency upgrades relative to the original “char GPT-2 in L3” sketch.

## P1 eval + C readahead (2026-08-03)

| Change | Detail |
|--------|--------|
| Gate instrumentation | `AgentTurn.gate_events`; battery `gate_stats`; product false-call **blocked 27/27** |
| Pure dual scores | emission-aware **72/101**, user-visible **99/101**, freeze pin **90/101** |
| Suite doc | [AGENT_BATTERY_SUITE.md](AGENT_BATTERY_SUITE.md) |
| KB on/off | lookup **20/20 → 0/20** empty table (`scripts/kb_onoff_ablation.py`) |
| SciQ multi-seed probe | limit 100 seeds 42/43/44: mean **60.7±1.2** (paper SciQ 66.5 is limit 400) |
| Prefetch knobs | `CLM_W_PREFETCH_*` in `ll3m_v2.c`; grid **no keep** |
| Multi RR | `multi_bench --burst K`; product multi remains pthread |
| Paper | **v0.7.9** gate decomp + KB note |

## Dense knowledge ladder (2026-07-21)

Full doc: [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md).

| Change | Detail |
|--------|--------|
| KB v2 | Canonical entries + aliases; ~471 entries / ~1164 keys in `kb_facts.json` |
| Lookup | Word-boundary matching; loaders accept v1 flat or v2 |
| Pretrain density | Synthetic 700M (kb-heavy) + FineWeb 500M; doc cap 1200 |
| Near-dedup | MinHash LSH Jaccard 0.85 in `prepare_base_english.py` |
| QAT lock | ternary + int8 asserted through continue + fact SFT + tool SFT |
| Eval | Std MC (HellaSwag, ARC-E, WinoGrande, BoolQ, OBQA, SciQ, COPA) vs Qwen3.5-0.8B NF4 |
| Ops | spark1 `ll3m-dense`; launch / poll / kill scripts |

## Architecture (v2)

| Before | After |
|--------|--------|
| LayerNorm | RMSNorm |
| Absolute `wpe` | RoPE (optional **partial** `rope_partial`) |
| GELU 4× MLP | SwiGLU (~8/3 d) |
| MHA | GQA |
| Linear biases | No bias on projections |
| — | QK-Norm |
| — | Optional Gemma **logit softcap** |
| — | PaLM **z-loss** train regularizer (`--z-loss`) |

Inspired by **Qwen3 dense** + LLaMA stack (not MoE / not Qwen3.5 hybrid).  
See [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) for the full inventory.

### Research tweaks considered (2024–2026)

- **Deep model + arXiv survey (canonical):** [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md)  
- **Architecture tweak map:** [RESEARCH.md](RESEARCH.md)  
- **Backlog:** `AGENTS.md` quant 1–7 + research R-B…R-J  

| Tweak | Decision |
|-------|----------|
| Partial RoPE (LLaMA 3.1) | Optional; default full for ckpt compat |
| z-loss (PaLM / OLMo) | Default `1e-4` (train only) |
| Shared-head MTP (DeepSeek-style signal) | Default `--mtp 0.1` (train only, no extra weights) |
| Peri-LN (Gemma2 / OLMo2) | Optional `--peri-ln` (failed on v75 width; LayerScale alone) |
| LayerScale | On for deep stacks (`1e-4` on v75) |
| Final logit softcap (Gemma 2) | Optional; QK-Norm already stabilizes attn; re-test under QAT (R-J) |
| Multi-stage data (SmolLM / Phi) | Partial (GPT-4 mix); curriculum still open (R-D) |
| Host tools / Engram-style knowledge | **Product path** (agent mode + lookup); not in weights |
| Native ternary QAT | **In flight** after v75 |
| MLA / MoE / SWA / GDN / Mamba hybrid | Deferred (C + L3 + short T) |
| Diffusion LM decode | Rejected for product C path |
| Huge vocab BBPE | Rejected (embed steals L3) |

## Tokenization and data

| Before | After |
|--------|--------|
| Char V=65 | Domain **BPE V=2048** |
| Tiny Shakespeare only | **TinyStoriesV2-GPT4** + Shakespeare style spice |
| ~80 MiB mixed GPT-3.5/4 slice | **~400 MiB GPT-4** train (+ full official valid) |
| Char-cut train/val | **Story-level** split + near-exact **dedup** |
| Encode at train start | Pre-encoded `.bin` token streams |
| Manual curl | `scripts/download_data.sh` + improved `prepare_data.py` |

## Training

| Before | After |
|--------|--------|
| CPU-only mindset | ROCm GPU + CPU |
| Full corpus on GPU | **Host corpus**; batch H2D only |
| Uncapped VRAM | **`--gpu-mem-fraction`** (default 0.45) |
| No QAT | **`--qat int8|ternary`** |
| Infer gate on train tensors | Gate on **C infer WS** |

## Serve quant and L3 residency

| Before | After |
|--------|--------|
| f32 or int8 | **Ternary 2-bit pack** default (`CLM_QUANT=2`) |
| Optional mlock | **mlock + prefault by default** |
| Soft “fits L3” print | **`--l3-resident`**: CCD bind, warm, refuse oversize |
| Informal perf | **`measure_cache.sh` → `L3_RESIDENCY_GATE`** |

See [L3_RESIDENCY.md](L3_RESIDENCY.md), [QUANTIZATION.md](QUANTIZATION.md).

## Still open

| Item | Status |
|------|--------|
| Full C forward for **CLM2 / v2** (RoPE, GQA, SwiGLU, BPE) | Export + WS accounting done; kernels pending |
| Eval suite tokenizer-aware probes for BPE | Partial (generate/chat handle BPE) |
| Ternary C for v2 exports | After v2 forward |
| Fresh v2 train on new GPT-4 mix | **Running:** `checkpoints/v2_dense` L18×d320 (~20.8M), then ternary QAT |
| Qwen3.5 GDN hybrid | Deferred |

## Key artifacts

| Path | Meaning |
|------|---------|
| `checkpoints/best_L6d192_infer.pt` | v1 quality archive (char Shakespeare) |
| `checkpoints/best_L6.clm` | C-export for L3 residency demos |
| `checkpoints/v2/` | v2 BPE train |
| `checkpoints/v2_ternary/` | ternary QAT fine-tune |
| `data/tokenizer.json`, `mix_*.bin` | BPE + tokens |
