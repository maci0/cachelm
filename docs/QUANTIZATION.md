# Quantization, QAT, ternary, and GPU VRAM

Serve path defaults to **ternary** weights in C. Train can stay fp32 or use **QAT**.
GPU training caps VRAM so the desktop stays usable.

## Why quantize

| Goal | How |
|------|-----|
| Fit more in L3 | int8 ≈ 4× denser than f32; ternary pack ≈ 4× denser than int8 for Linears |
| Match C logits | QAT trains through the same grid |
| Bandwidth | Fewer bytes per GEMV |

Master weights in PyTorch remain **fp32**. Activations and KV stay **f32** today.

## Post-training quant in C (always at load for CLM1)

### Product densest packing (default build)

| Tensor | Format | Bits / weight | Notes |
|--------|--------|---------------|--------|
| **Linear W** | ternary 2-bit pack | **2** | BitNet-style `{-1,0,1}` + per-row absmean scale |
| **wte / LM head** | **int8** + per-row scale | **8** | Tied embeds; biggest remaining win vs f32 table |
| Norms / LayerScale | f32 | 32 | Tiny vs Linears |
| wpe (v1 only) | f32 | 32 | `T·d` only |
| Activations / KV | f32 | 32 | Not weights; GQA shrinks KV |

Ternary is already information-theoretically near-optimal for 3-level weights
(true trit packing ≈1.58 bit is rarely worth the decode cost; we use 2-bit bytes).
**Do not** ship pure 1-bit binary without quality re-proof under QAT. Research scaffold exists
(`--qat binary`, `--quant binary` → **L3B2**, C load+scalar GEMV); product default remains ternary.
See `docs/L3_CAPACITY_SEARCH.md` for reinvest ~142M under gate.

### Ternary Linears (`-DCLM_QUANT=2`, default)

Per row:

```
scale = mean(|W|)
q     = clamp(round(W / scale), -1, 1)
```

Packed as **2 bits/weight** (4 weights/byte): codes `0→-1, 1→0, 2→+1`.  
GEMV: `y = b + scale * sum(q_i * x_i)` with zero-skip.

### int8 Linears (`-DCLM_QUANT=1`)

Per-row absmax → [-127, 127]; AVX-512 VNNI path when available.

### Embed int8 (`-DCLM_EMBED_QUANT=1`, default when ternary)

Per-token-row absmax → int8 codes + `float scale[V]`.  
Lookup / LM head dequant on the fly (no f32 embed table in the arena).

```bash
cd c_infer
make                                    # ternary Linears + int8 embeds (product)
make CFLAGS='... -DCLM_QUANT=2 -DCLM_EMBED_QUANT=0'   # ternary Linears, f32 embeds
make CFLAGS='... -DCLM_QUANT=1 -DCLM_EMBED_QUANT=0'   # int8 Linears
```

### Embed QAT experiment (int8 vs ternary, trained)

Same base (`v2_dense_ternary`, Linear ternary QAT), +4k steps, `--embed-qat`:

| Arm | Flags | Held-out val CE | Notes |
|-----|-------|-----------------|--------|
| Baseline | Linear ternary, f32 embeds | **1.484** | `v2_dense_ternary/best.pt` |
| **int8 embeds** | `--qat ternary --embed-qat int8` | **1.478** | best quality; product default |
| ternary embeds | `--qat ternary --embed-qat ternary` | **1.571** | recovers from post-hoc collapse (~2.6) but still worse |

ΔCE ternary−int8 after QAT ≈ **+0.09** (~6% relative). Density win for ternary embeds is only ~0.5 MiB on L18×d320. **Keep int8 embeds.**

Train: `bash scripts/run_embed_qat_compare.sh` then `python scripts/compare_embed_qat.py`.

## QAT (quantization-aware training)

### Mechanism

Fake-quant weights in forward; STE so `round` is identity in backward.
Optimizer still updates fp32 parameters.

### CLI

```bash
--qat none      # default
--qat int8      # absmax fake-quant
--qat ternary   # BitNet-style absmean + round to {-1,0,1}
```

Implementation: `ll3m/qat.py` (`enable_qat_on_model`, pack helpers).

### Recipe (product: dense ternary)

**Do not** train ternary from random init for the whole base run. Early STE noise
hurts language learning. **Do** keep ternary QAT on for every stage after you
switch (tool SFT, polish) so serve matches the train grid.

| Phase | Mode | Goal |
|-------|------|------|
| 1 Base | `--qat none` | Max capability on GPT-4 mix |
| 2 Serve adapt | `--qat ternary` | Long finetune on same mix, lower LR |
| 3 Tools (later) | `--qat ternary` | Still on the grid |
| 4 Export | C pack | Real 2-bit weights |

One-shot pipeline (L18×d320, ~20.8M, ternary WS ~15 MiB ≪ 32 MiB CCD):

```bash
bash scripts/train_dense_ternary.sh
# override: STEPS1=25000 STEPS2=12000 LAYERS=18 EMBD=320
```

Manual phase 2 example:

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m ll3m.train \
  --resume checkpoints/v2_dense/best.pt \
  --qat ternary --serve-quant ternary \
  --n-layer 18 --n-embd 320 --n-head 8 --n-kv-head 2 \
  --peri-ln --layer-scale-init 1e-2 \
  --batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.40 \
  --lr 8e-5 --steps 37000 --device rocm \
  --out-dir checkpoints/v2_dense_ternary
```

Expect a short **loss shock** when QAT turns on; it should recover.

L3 gate uses `--serve-quant ternary` by default (true product packing), not int8.

## Ternary vs int8 (decision guide)

| | int8 | ternary |
|--|------|---------|
| L3 density | Good | **Best** for Linears |
| C path (v1) | VNNI-friendly | Packed GEMV (default build) |
| Fidelity without QAT | Better | Worse (garbled samples more likely) |
| With QAT | Strong default | Worth it if L3-bound on weights |
| KV / acts | Unchanged | Unchanged (use GQA for KV) |

## GPU VRAM limits (train)

System freezes were mitigated by:

| Control | Default | Effect |
|---------|---------|--------|
| `--gpu-mem-fraction` | **0.45** | Hard process VRAM cap |
| Corpus location | **CPU + pin_memory** | Only microbatches on GPU |
| `--batch-size` | **8** | Lower activation peak |
| `--grad-accum` | **8** | Effective batch without VRAM |
| Grad checkpoint | **on** | Recompute activations |

If the dGPU drives a display, prefer **0.35–0.40**.

Safe pattern:

```bash
--batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.40 --device rocm
```

## Density numbers (L6d192 C)

| Quant | Weights | Infer WS |
|-------|---------|----------|
| int8 | ~2.81 MiB | ~5.25 MiB |
| ternary | ~0.91 MiB | ~3.35 MiB |

## Roadmap

1. VRAM-safe train (done)  
2. Ternary C default + QAT flag (done)  
3. C **v2** forward + L3T2 `.ll3m` pack (done; RoPE train–serve match)  
4. **v75 ternary QAT + serve-grid compare** (in flight: waiter → `v75_ternary`)  
5. Train–serve grid lock / first-class `BitLinear` (backlog)  
6. Optional: W1.58A8 activations, TL-style ternary GEMV if still CPU-bound  

Active backlog lives in **`AGENTS.md`** (quant + research follow-ups).

## Research sources (ternary)

Documented with IDs and transfer notes in
[RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) §1.12 and §2.1:

| Source | Why it matters here |
|--------|---------------------|
| BitNet / b1.58 / 2B4T | Native ternary STE, absmean scales |
| bitnet.cpp (TL, I2_S) | Edge kernels + **train-matched lossless** pack |
| Tequila, TernaryLM, CAT-Q | QAT vs PTQ recipes; absmean default in open ternary |
| T-SAR | CPU SIMD ternary; TLUT bandwidth caveats |

**Stance:** QAT on the same grid as C L3T2 ≫ post-hoc ternarize. Do not fork the
architecture into “BitNet blocks”; quant + kernels only.

## Code map

| Path | Role |
|------|------|
| `ll3m/qat.py` | STE + ternary pack/unpack |
| `ll3m/train.py` | `--qat`, `--gpu-mem-fraction` |
| `scripts/export_weights.py` | fp32/QAT → `.ll3m` L3T2 |
| `scripts/compare_fp32_vs_qat.py` | serve-grid quality compare |
| `scripts/run_v75_ternary_qat_when_ready.sh` | auto QAT after v75 |
| `c_infer/ll3m.c` / `ll3m_v2.c` | Load-time quant + GEMV |
| `c_infer/Makefile` | `-DCLM_QUANT=2` default |
