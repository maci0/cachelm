# Architecture v2

Default trainable stack after the capability upgrade. Legacy GPT-2-style models
use `arch="v1"`.

## Goals

| Goal | Approach |
|------|----------|
| Quality per param | Qwen3-dense-style block |
| English per context step | Domain BPE V≈2048 |
| Fit + stay in L3 | GQA (small KV) + ternary/int8 Linears |
| Train freely | PyTorch CPU/ROCm; WS gate is **infer** only |

## Block diagram (`arch=v2`)

```
token ids
  → wte Embedding  (tied LM head)
  → Dropout
  → for each layer:
        x = x + Attn( RMSNorm(x) )
            Attn: Q,K,V Linear (no bias)
                  QK-Norm (RMSNorm over head_dim)
                  RoPE on Q,K
                  GQA (repeat KV heads to match Q)
                  causal SDPA
                  O Linear (no bias)
        x = x + SwiGLU( RMSNorm(x) )
            SwiGLU: gate, up, down (no bias)
  → RMSNorm
  → logits = linear(x, wte.weight)
```

### Hyperparameters

| Field | Meaning | Example |
|-------|---------|---------|
| `n_layer` | Depth | 10 |
| `n_embd` | Width | 320 |
| `n_head` | Query heads | 8 |
| `n_kv_head` | KV heads (GQA) | 2 |
| `intermediate_size` | SwiGLU mid | ~896 (≈ 8/3 · d, 64-aligned) |
| `block_size` | Context T | 256 |
| `vocab_size` | BPE size | 2048 |
| `rope_theta` | RoPE base | 10000 |
| `rope_partial` | Fraction of head_dim with RoPE | 1.0 (try 0.5 LLaMA-3.1 style) |
| `logit_softcap` | Gemma final softcap; 0=off | 0 (try 30) |
| `peri_ln` | Post-residual RMSNorm (Gemma/OLMo) | False (`--peri-ln`) |
| `layer_scale_init` | Residual LayerScale init; 0=off | 0 (try `1e-2` if deep) |
| `dropout` | Train only | 0.15 |

Train-only (not extra serve weights): `--z-loss` (default `1e-4`), `--mtp` shared-head multi-token aux (default `0.1`). See [RESEARCH.md](RESEARCH.md).

`GPTConfig` lives in `ll3m/config.py` (`to_dict` / `from_dict` for checkpoints).

## Qwen3 / LLaMA inventory (what we took)

| Feature | Source | Status in ll3m v2 |
|---------|--------|----------------------|
| RMSNorm (pre-norm) | LLaMA, Qwen | Yes |
| RoPE (no absolute `wpe`) | LLaMA, Qwen | Yes |
| SwiGLU MLP ~8/3 d | LLaMA, Qwen | Yes |
| GQA | LLaMA-2/3, Qwen | Yes |
| QK-Norm | Qwen3 | Yes |
| No Linear biases | LLaMA, Qwen | Yes |
| Tied embed / LM head | common small-LM | Yes |
| Partial RoPE | LLaMA 3.1 | Optional (`rope_partial`) |
| z-loss | PaLM (also used in modern stacks) | Train (`--z-loss`) |
| Final logit softcap | Gemma 2 | Optional (`logit_softcap`) |
| MoE / huge V / GDN hybrid | Qwen3 MoE / Qwen3.5 | **No** (L3 / C cost) |

## Tokenizer

- Implementation: `ll3m/tokenizer.py` (pure Python byte-level BPE)  
- Artifact: `data/tokenizer.json`  
- Train load: prefer `data/mix_train.bin` / `mix_val.bin` (uint16 ids)  

Why V=2048 not 32k: embedding cost is `V · d` floats; large V steals L3 from layers.

## Data pipeline

```bash
# Prefer GPT-4 TinyStories (~1.5 GiB partial by default; MAX_BYTES=0 for full ~2.2 GiB)
bash scripts/download_data.sh

# Story-level split, quality filters, dedup, official val, BPE, .bin tokens
uv run python scripts/prepare_data.py --vocab-size 2048 --bpe-train-chars 40000000
```

Pipeline details (`prepare_data.py`):

- Auto-picks `TinyStoriesV2-GPT4-*.txt` over older mixed GPT-3.5/4 slices
- Splits on `<|endoftext|>` / blank lines (story-level holdout, no char-cut leakage)
- Drops incomplete last story on range-partial downloads
- Quality filters: length, printable/alpha, word/bigram spam, sentence end
- Near-exact dedup + train/val cross-dedup (whitespace-normalized SHA1)
- Uses official GPT-4 valid when present (pure stories, no Shakespeare)
- Shakespeare interleaved as style spice (~5% default)
- Stratified BPE train sample (40M chars default)

Outputs: mix texts, tokenizer, binary token streams, `mix_manifest.json`.

## Training recipe (VRAM-safe)

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m ll3m.train \
  --arch v2 --tokenizer-mode bpe \
  --n-embd 320 --n-layer 10 --n-head 8 --n-kv-head 2 --block-size 256 \
  --batch-size 8 --grad-accum 8 \
  --gpu-mem-fraction 0.40 \
  --device rocm --infer-scope ccd \
  --steps 20000 --dropout 0.15 --lr 3e-4 \
  --prompt "Once upon a time" \
  --out-dir checkpoints/v2
```

Ternary QAT fine-tune (after a good fp32/v2 best):

```bash
.venv/bin/python -m ll3m.train \
  --resume checkpoints/v2/best.pt --qat ternary \
  --batch-size 8 --grad-accum 8 --gpu-mem-fraction 0.40 \
  --lr 1e-4 --steps 8000 --device rocm \
  --out-dir checkpoints/v2_ternary
```

## Infer working set (v2)

`estimate_infer_ws()` models int8-style packing for Linears + GQA KV. Ternary C
packing is denser than that estimate (good: estimate is conservative for the
gate). Always check C printed `infer WS` for the real serve footprint on v1;
v2 C forward is still landing.

Example (approx, int8 accounting):

| Config | Params | Infer WS est. |
|--------|--------|----------------|
| L6 d256 kv2 T256 V2k | ~6M | ~12 MiB |
| L10 d320 kv2 T256 V2k | ~11.8M | ~19 MiB |
| L12 d384 kv3 T256 V2k | ~19M | ~29 MiB |

## Export

```bash
.venv/bin/python scripts/export_weights.py \
  --checkpoint checkpoints/v2/best.pt \
  --tokenizer data/tokenizer.json \
  --out checkpoints/v2/best.clm
```

- **CLM1:** v1 char (full C path)  
- **CLM2:** v2 + tokenizer blob (export ready)

## v1 legacy

```bash
.venv/bin/python -m ll3m.train \
  --arch v1 --tokenizer-mode char \
  --data data/tinyshakespeare.txt \
  --n-embd 192 --n-layer 6 --n-head 6 --block-size 128 \
  ...
```

Archive quality peak: `checkpoints/best_L6d192_infer.pt`.

## Not copied from Qwen3.5

MoE, 150k BBPE, Gated DeltaNet hybrid, native VL, thinking-mode product surface.
Those fight the L3 + simple C story at this scale.

## Code map

| Path | Role |
|------|------|
| `ll3m/config.py` | Config + `n_params` |
| `ll3m/model_torch.py` | v1/v2 modules, WS estimates |
| `ll3m/tokenizer.py` | BPE |
| `ll3m/train.py` | CLI |
| `ll3m/qat.py` | QAT STE |
| `scripts/prepare_data.py` | Mix + BPE + bins |
| `scripts/export_weights.py` | CLM1/CLM2 |
