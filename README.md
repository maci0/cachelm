# LL3M

**LL3M** (*ell-ell-three-em*): an **LLM** that lives in **L3**.

Small decoder-only language model trained **from scratch**, with one hard product rule:


> At **inference time**, the model’s true working set (weights + activations + KV
> cache) must **fit in one CPU L3 CCD** and stay there on the hot path: **no DRAM
> round-trips** for that data after pin/warm. Proof is bare-metal **C** + `perf`,
> not Python RSS.

Training may use full DRAM or a ROCm GPU. The L3 claim is about **serving only**.

**Default model stack (v2):** RMSNorm, RoPE, SwiGLU, GQA, QK-Norm (Qwen3/LLaMA
dense lessons), optional partial RoPE / logit softcap / z-loss, small domain
**BPE (V≈2048)**, data mix **TinyStoriesV2-GPT4 + Shakespeare**.  
**Default serve quant:** **ternary** 2-bit packed Linears (`-DCLM_QUANT=2`).  
Legacy char GPT-2 models remain as `arch=v1` / int8 optional.

### Docs

| Doc | Topic |
|-----|--------|
| [docs/README.md](docs/README.md) | Index |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Why this project, data, math |
| [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) | v2 architecture + tokenizer + data |
| [docs/RESEARCH_MODELS_2026.md](docs/RESEARCH_MODELS_2026.md) | **Deep model + arXiv survey → LL3M actions** (2026-07-17) |
| [docs/NATIVE_TERNARY_PIPELINE.md](docs/NATIVE_TERNARY_PIPELINE.md) | Research → native ternary train + tools stages |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Architecture tweaks map (earlier pass) |
| [docs/TOOL_CALLING.md](docs/TOOL_CALLING.md) | Tool calling design (agent loop + SFT) |
| [docs/L3_RESIDENCY.md](docs/L3_RESIDENCY.md) | L3-only residency protocol + perf gate |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | int8, ternary, QAT, GPU VRAM caps |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Active train / QAT / research follow-ups |
| [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) | What changed and why |
| [AGENTS.md](AGENTS.md) | Ops, freeze, quant + research backlog |
| [c_infer/README.md](c_infer/README.md) | C runtime build / run |

---

## Table of contents

1. [What this is](#what-this-is)
2. [Why L3 cache (and no DRAM)](#why-l3-cache-and-no-dram)
3. [Hardware target](#hardware-target)
4. [Dataset](#dataset)
5. [Model](#model)
6. [Training](#training)
7. [C inference (the cache claim)](#c-inference-the-cache-claim)
8. [Measuring capability](#measuring-capability)
9. [Repository layout](#repository-layout)
10. [Quick start](#quick-start)
11. [Configs and results](#configs-and-results)
12. [What we do not claim](#what-we-do-not-claim)

---

## What this is

| Piece | Role |
|--------|------|
| **Model** | Decoder-only transformer (v2 default; v1 legacy) |
| **Tokens** | Small domain **BPE V=2048** (or char V=65 for v1) |
| **Task** | Next-token LM on TinyStories-heavy mix (+ Shakespeare style) |
| **Train** | PyTorch on CPU or **ROCm** GPU; Lion; optional **QAT** (int8/ternary) |
| **Serve** | Portable **C** (`c_infer/`): ternary or int8 Linears, KV cache, L3 pin |
| **Constraint** | Infer WS ≤ one L3 CCD; DRAM share of L2 fills ≈ noise after warm |

| Checkpoint family | Params (order) | C infer WS (order) | Notes |
|-------------------|----------------|--------------------|--------|
| v1 L6×d192 char (archive) | ~2.7M | ~3.4 MiB ternary / ~5.3 MiB int8 | Best pure-Shakespeare NLL historically |
| v2 L10×d320 GQA BPE | ~11.8M | ~19 MiB (int8 estimate) | Primary capacity path; ternary denser still |
| v2 + ternary QAT | same | smaller weight arena in C | Fine-tune for serve quant |

---

## Why L3 cache (and no DRAM)

```
registers → L1 → L2 → L3 → DRAM
  ~1 c       few   tens  hundreds of cycles
```

If the **entire** hot data set for decode lives in L3:

- L2 misses are filled from **local L3**, not DRAM  
- Latency is compute + on-chip bandwidth  
- You can **prove** it with AMD `l2_fill_rsp_src.*` counters  

That requires:

1. Explicit sizing of **weights + scratch + KV**  
2. Contiguous arenas, **mlock**, prefault, warm  
3. Bind to **one CCD** (this chip has 2×32 MiB L3)  
4. A C process without multi-GiB PyTorch RSS  

**Training-in-L3 is not the goal.** Train may use hundreds of MiB of DRAM/VRAM.

Full protocol: [docs/L3_RESIDENCY.md](docs/L3_RESIDENCY.md).

---

## Hardware target

Development host (document numbers against this):

| Resource | Size |
|----------|------|
| CPU | AMD Ryzen 9 9950X |
| L1d / core | 48 KiB |
| L2 / core | 1 MiB |
| **L3 per CCD** | **32 MiB** |
| L3 chip total | 64 MiB (2 CCDs) |
| Train GPU (optional) | RX 7900 XTX (24 GiB), ROCm |

**Default product gate:** 95% of **one CCD** ≈ **30.4 MiB** C infer WS.  
**Residency gate:** WS ≤ CCD L3 **and** DRAM share of L2 fills ≤ ~5% after `--l3-resident`.

---

## Dataset

| File | Role |
|------|------|
| `data/TinyStoriesV2-GPT4-train.partial.txt` | GPT-4 TinyStories (~400 MiB default download) |
| `data/TinyStoriesV2-GPT4-valid.txt` | Official GPT-4 holdout (~22 MiB) |
| `data/tinyshakespeare.txt` | Public-domain Shakespeare (~1.1 MB, style spice) |
| `data/mix_train.txt` / `mix_val.txt` | Story-level mix (~8% Shakespeare in train) |
| `data/mix_*.bin` | Pre-encoded BPE token ids (uint16) |
| `data/tokenizer.json` | Domain BPE, **V=2048** |

Prepare (download GPT-4 slice + build mix / BPE / bins):

```bash
bash scripts/download_data.sh          # MAX_BYTES=0 for full ~2.2 GiB
uv run python scripts/prepare_data.py --vocab-size 2048
```

Story-level split + dedup; val stays pure stories when official valid is present.
Tiny Shakespeare alone remains available for v1 char experiments. Details:
[docs/OVERVIEW.md](docs/OVERVIEW.md), [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md).

---

## Model

### v2 (default)

```
ids → Embedding (tied LM head) → Dropout
    → ×N:  x += Attn(RMSNorm(x))   # GQA, RoPE, QK-Norm, no bias
           x += SwiGLU(RMSNorm(x))
    → RMSNorm → logits = h @ wte.T
```

### v1 (legacy)

GPT-2-style: absolute `wpe`, LayerNorm, MHA, GELU 4× MLP, biases. Char vocab.

### Serve quantization

| Mode | Build | Linear storage |
|------|--------|----------------|
| **Ternary** (default) | `-DCLM_QUANT=2` | ~2 bits/weight packed |
| int8 | `-DCLM_QUANT=1` | 8-bit + VNNI path |

Embeddings and norms stay f32. See [docs/QUANTIZATION.md](docs/QUANTIZATION.md).

---

## Training

| Knob | Default / note |
|------|----------------|
| Optimizer | Lion (AdamW optional) |
| Device | `--device auto\|cpu\|rocm` |
| GPU VRAM | **`--gpu-mem-fraction 0.45`**; corpus on **CPU** (batches only on GPU) |
| Microbatch | Prefer **8** + higher `--grad-accum` on GPU |
| Grad checkpoint | On by default |
| QAT | `--qat none\|int8\|ternary` |
| Infer gate | Refuse configs whose C WS exceeds L3 (unless `--force`) |

ROCm note: `uv sync` may reinstall CUDA wheels. Re-pin if needed:

```bash
uv pip install torch==2.10.0+rocm7.0 --index-url https://download.pytorch.org/whl/rocm7.0
```

---

## C inference (the cache claim)

### Build and L3-resident run

```bash
cd c_infer && make
ulimit -l unlimited   # if mlock fails

./ll3m_c -m ../checkpoints/best_L6.clm \
  -p $'ROMEO:\n' -n 200 --greedy \
  --l3-resident --warm 12

# Hardware proof (DRAM share + WS fit)
make cachestat
```

### What `--l3-resident` does

1. Bind process to **one L3 CCD**  
2. **mlock** weight + state arenas  
3. **Prefault** every cache line  
4. **Warm** full forwards so L3 holds the set  
5. **Refuse** if infer WS > CCD L3  

### Measured example (L6 ternary on this host)

```
infer WS:     3.35 MiB / 32 MiB CCD
mlock:        model=locked state=locked
CCD bind:     OK
L2 fills:     ~97% local L3, ~3% DRAM
L3_RESIDENCY_GATE: PASS
```

Formats: **CLM1** = v1 char (C fully supported). **CLM2** = v2 export + tokenizer blob
(export ready; C v2 forward still catching up—size accounting matches packing).

---

## Measuring capability

| Axis | Metric |
|------|--------|
| Compression | Val NLL, **BPC**, PPL |
| Domain | Shakespeare-like vs other probes (char eval suite) |
| Ranking | Top-1 / top-5 |
| Format / diversity | Samples look like the corpus |
| **Cache** | C infer WS + `L3_RESIDENCY_GATE` |

```bash
uv run python -m ll3m.eval --checkpoint checkpoints/best_L6d192_infer.pt --budget-scope ccd
```

BPE models need eval updates for tokenizer-aware probes; use generate/chat with
`tokenizer_path` in the checkpoint for qualitative checks.

---

## Repository layout

```
ll3m/
  ll3m/
    config.py          # GPTConfig v1/v2
    model_torch.py     # Trainable GPT, RoPE/GQA/SwiGLU, WS estimates
    model.py           # NumPy v1 reference
    tokenizer.py       # Domain BPE
    train.py           # Training CLI (VRAM + QAT)
    qat.py             # Fake-quant STE (int8 / ternary)
    eval.py / generate.py / chat.py  # generate/chat default to C engine
    c_engine.py        # thin subprocess wrapper around ll3m_c
  c_infer/
    ll3m.c/.h          # Load, quant (ternary/int8), forward, pin/warm
    ll3m_v2.c          # CLM2 / v2 f32 decode (RoPE, MQA, SwiGLU)
    main.c             # --l3-resident / --stream / --token-ids CLI
    measure_cache.sh   # perf DRAM gate
  scripts/
    prepare_data.py    # Mix + BPE + .bin tokens
    export_weights.py  # CLM1 / CLM2
  data/                # corpora, tokenizer, bins
  checkpoints/         # .pt, .clm, train logs
  docs/                # this documentation set
  tests/
```

---

## Quick start

```bash
# 1) env
uv sync
uv pip install torch==2.10.0+rocm7.0 --index-url https://download.pytorch.org/whl/rocm7.0

# 2) data + BPE (GPT-4 TinyStories + official val)
bash scripts/download_data.sh
uv run python scripts/prepare_data.py --vocab-size 2048

# 3) train v2 (VRAM-safe GPU)
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m ll3m.train \
  --arch v2 --tokenizer-mode bpe \
  --n-embd 320 --n-layer 10 --n-head 8 --n-kv-head 2 --block-size 256 \
  --batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.40 \
  --device rocm --infer-scope ccd \
  --steps 20000 --dropout 0.15 --lr 3e-4 \
  --prompt "Once upon a time" \
  --out-dir checkpoints/v2

# 4) optional ternary QAT fine-tune
.venv/bin/python -m ll3m.train \
  --resume checkpoints/v2/best.pt --qat ternary \
  --batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.40 \
  --lr 1e-4 --steps 8000 --device rocm \
  --out-dir checkpoints/v2_ternary

# 5) C L3-resident serve (v1 CLM1 path fully supported)
.venv/bin/python scripts/export_weights.py \
  --checkpoint checkpoints/best_L6d192_infer.pt \
  --out checkpoints/best_L6.clm
cd c_infer && make
./ll3m_c -m ../checkpoints/best_L6.clm -p $'ROMEO:\n' -n 200 \
  --greedy --l3-resident
make cachestat
```

---

## Configs and results

### v1 L6×d192 (char Shakespeare archive)

| Metric | Approx. value |
|--------|----------------|
| Params | 2.7M |
| Val NLL (fp32) | ~1.46 |
| C WS ternary | **~3.4 MiB** |
| C WS int8 | ~5.3 MiB |
| L3 residency | **PASS** (DRAM fill share ~3%) |

### v2 L10×d320 GQA (BPE mix)

| Metric | Approx. value |
|--------|----------------|
| Params | ~11.8M |
| C infer WS (est. int8 packing) | ~19 MiB |
| Train | ROCm; corpus on host; VRAM fraction 0.40 |
| Ternary QAT | Fine-tune from fp32 best (`checkpoints/v2_ternary`) |

Numbers move with training; check `checkpoints/*/progress.json` and `meta.json`.

### Inference speed order of magnitude (C, L6)

Depends on quant and CPU. Ternary is denser; int8+VNNI can be faster in tok/s.
Always re-bench after quant changes.

---

## What we do not claim

- That **training** fits in L3  
- That **Python RSS** fits in L3  
- **Absolute zero** DRAM counters including process start and `perf` attach  
- Reasoning, tools, instruction following, world knowledge  
- Bit-exact match between quantized C and fp32 PyTorch  
- Full C **v2** forward parity yet (export + WS accounting yes; kernels in progress)

---

## Design principles

1. **Honest constraint:** size, pin, warm, and measure on-chip fills.  
2. **Train anywhere, serve in L3.**  
3. **Density for residency:** ternary / int8 + GQA free L3 for capability.  
4. **Capability matches the corpus:** compress and continue the mix well.  
5. **One path:** prepare data → train → export → `ll3m_c --l3-resident` → `cachestat`.
