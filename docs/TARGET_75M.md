# Target model: max params in 30.4 MiB L3 (product freeze)

**FROZEN product shape: L23 × d576, h12/kv1, mid=1536 → 78,815,712 params.**  
Train recipe v2: **no peri-LN**, LayerScale **1e-4**, lr 3e-4, patience 0.  
Train: `checkpoints/v75` → QAT `checkpoints/v75_ternary`. Record: `checkpoints/v75_freeze.json`.

## Gate (product)

| Item | Value |
|------|--------|
| CCD L3 | **32.00 MiB** (this machine) |
| Product budget | **95% CCD = 30.40 MiB** (`31,876,710` bytes) |
| Arch | v2 |
| Vocab / T | 2048 BPE / 256 |
| peri_ln / LayerScale | **False** / **`1e-4`** (peri-LN blocked early learning on this shape) |
| Serve | Linear **ternary 2-bit** + embed **int8** |
| Estimator | `estimate_infer_ws(cfg, quant="ternary", embed_quant="int8")` |

Search objective: **maximize `n_params` under gate**, tie-break deeper, then wider, then lower WS.
Formula params match `sum(p.numel() for p in GPT(cfg).parameters())` for all champions below.

---

## Exact best shapes (deterministic grid)

### A. Product pick (recommended)

Standard heads `{8,12,16}`, **default** SwiGLU mid (`≈8/3 d`, 64-aligned), practical `head_dim ∈ {32,40,48,64}`, `kv ∈ {1,2}`.

| Field | Value |
|-------|--------|
| **n_layer** | **23** |
| **n_embd** | **576** |
| **n_head / n_kv_head** | **12 / 1** (MQA) |
| intermediate | **1536** (default) |
| head_dim | 48 |
| **n_params** | **78,815,712** (with peri would be 78,842,208) |
| **Infer WS** | **30.22 MiB** (99.4% fill) |

```bash
--n-layer 23 --n-embd 576 --n-head 12 --n-kv-head 1 --block-size 256 \
  --layer-scale-init 1e-4
# no --peri-ln; intermediate left default (=1536)
```

Beats the old L27×d512 shape on params while still filling the L3 gate.

### B. Absolute max params (no style constraints)

Any `n_head` 4..32, mid swept in steps of 64, `head_dim` 16..128, embd step 8.

| Field | Value |
|-------|--------|
| **n_layer** | **37** |
| **n_embd** | **408** |
| **n_head / n_kv_head** | **12 / 1** |
| intermediate | **1472** (inflated vs default 1088) |
| head_dim | 34 |
| **n_params** | **80,937,884** |
| **Infer WS** | **30.3628 MiB** (99.88% fill) |

```bash
--n-layer 37 --n-embd 408 --n-head 12 --n-kv-head 1 \
  --intermediate-size 1472 --peri-ln --layer-scale-init 1e-2
```

Odd width / head_dim; train/C kernels may be less friendly. Use only if you want every last parameter.

### C. Practical mid-free (standard heads + classic head dims)

`h ∈ {8,12,16}`, `hd ∈ {32,40,48,64}`, mid free:

| Field | Value |
|-------|--------|
| **n_layer** | **41** |
| **n_embd** | **384** |
| **n_head / n_kv_head** | **12 / 1** |
| intermediate | **1408** |
| head_dim | 32 |
| **n_params** | **80,485,568** |
| **Infer WS** | **30.2072 MiB** (99.37% fill) |

Very deep (L=41). Prefer **A** unless you specifically want depth + fat FFN.

---

## Why not the old L27×d512

| Shape | params | WS MiB | fill |
|-------|-------:|-------:|-----:|
| **A L23 d576 h12/1 mid1536** | **78,842,208** | **30.317** | **99.7%** |
| Old L27 d512 h8/1 mid1408 | 75,453,312 | 29.300 | 96.4% |
| L22 d576 h12/1 mid1536 | 75,465,600 | 29.347 | 96.5% |
| L28 d512 h8/1 mid1408 | 78,209,024 | 30.143 | 99.2% |
| L32 d480 h12/1 mid1280 | 76,035,040 | 29.294 | 96.4% |

MQA (`kv=1`) wins max-params under this gate: `kv=2` burns activation/KV bytes and caps params ~8–9M lower on the same grid.

---

## Train recipe (product pick A)

```bash
# data already rebuilt → data/mix_manifest.json (~466M train tok)
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m ll3m.train \
  --arch v2 --tokenizer-mode bpe \
  --n-layer 23 --n-embd 576 --n-head 12 --n-kv-head 1 --block-size 256 \
  --layer-scale-init 1e-4 \
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
  --batch-size 4 --grad-accum 16 --gpu-mem-fraction 0.45 \
  --dropout 0.1 --lr 3e-4 --warmup 2000 --steps 50000 --patience 0 \
  --device rocm --infer-scope ccd --serve-quant ternary \
  --qat none \
  --prompt "Once upon a time" \
  --out-dir checkpoints/v75

# ternary Linear QAT + int8 embed QAT
.venv/bin/python -m ll3m.train \
  --resume checkpoints/v75/best.pt \
  --qat ternary --embed-qat int8 \
  --batch-size 4 --grad-accum 16 --gpu-mem-fraction 0.45 \
  --lr 5e-5 --steps $((STEP+8000)) --patience 0 \
  --device rocm --serve-quant ternary \
  --out-dir checkpoints/v75_ternary
```

Lower microbatch if VRAM pressure; keep effective batch via `--grad-accum`.

## Data

See `data/mix_manifest.json`: ~1.95M GPT-4 stories, **466,487,919** train tokens, 6.0M val tokens, V=2048 BPE, 5% Shakespeare spice.
