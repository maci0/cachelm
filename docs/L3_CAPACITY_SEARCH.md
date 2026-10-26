# L3 capacity search (reinvest denser packs into params)

**Date:** 2026-08-05  
**Gate:** 95% of one L3 CCD = **30.40 MiB** (this host 32 MiB instance)  
**Tool:** `scripts/search_l3_capacity.py` → `checkpoints/l3_capacity_search.json`  
**Estimator:** `estimate_infer_ws()` (norms + scales + packed Linears + embeds + acts/KV)

## Thesis

Under a fixed CCD budget, denser Linear packing frees **weight bytes**. Those bytes can be reinvested as **more parameters** (deeper/wider shape), not left as idle headroom.

You are not on a ternary computer: ternary/binary are **pack formats** for L3 bandwidth. Quality still depends on the alphabet and training.

## Packs compared

| Linear pack | Bits / weight | Embed default | Product C path |
|-------------|---------------|---------------|----------------|
| **ternary** `{-1,0,1}` | 2-bit + pad | int8 | **Yes** (L3T2) |
| **binary** `{-1,+1}` | 1-bit + pad | int8 | **No** (search estimate only) |

Same freeze shape L23×d576 mid1536 T=256:

| Pack | Infer WS | Saved vs ternary |
|------|----------|------------------|
| ternary+int8 | **30.22 MiB** (99.4% gate) | — |
| binary+int8 | **20.51 MiB** | **~9.7 MiB** |

## Results (T=256, V=2048, MQA, peri off, ls=1e-4)

### Product baseline

| Shape | Params | WS | Pack |
|-------|--------|-----|------|
| L23×d576 h12/1 mid1536 | **78,815,712** | 30.22 MiB (99.4%) | ternary+int8 |

### Densest ternary+int8 (ship pack)

Max found: **~80.7M** (+**2.4%** vs freeze).

| Rank | Shape | Params | WS | Δ vs freeze |
|------|-------|--------|-----|-------------|
| 1 | L38×d384 h8/1 mid1536 | 80,696,256 | 30.29 MiB | +1.88M |
| 2 | L38×d384 h12/1 mid1536 | 80,228,096 | 29.94 MiB | +1.41M |
| 3 | L35×d480 h12/1 mid1216 | 79,811,920 | 30.35 MiB | +1.00M |
| … | freeze L23×d576 | 78,815,712 | 30.22 MiB | 0 |

**Conclusion:** at **T=256** the ternary product pack is **already near the param ceiling**. Reinvesting under ternary alone buys only a couple percent params. Not worth a full retrain solely for +2%.

### Densest binary+int8 (research reinvest)

**Estimator gate 30.4** (old rule, overstates activations): max **~142–145M**.

| Rank | Shape | Params | WS est | Δ vs freeze |
|------|-------|--------|--------|-------------|
| 1 | L30×d576 h12/1 mid2304 | **142,257,024** | 30.22 MiB | +63.4M |
| 2 | L38×d512 h16/1 mid2048 | 141,835,136 | 30.21 MiB | +63.0M |
| 3 | L33×d576 h12/1 mid2048 | 141,766,560 | 30.23 MiB | +63.0M |

**C-calibrated gate** (ll3m_c smoke is authority; act scale ~0.39× est): max **~197M** under C-like ≤ 30.4 MiB.

| Rank | Shape | Params | C-like | Notes |
|------|-------|--------|--------|-------|
| 1 | L49×d480 mid2432 | **197,144,240** | ~30.2 | densest grid |
| 2 | L34×d576 mid2880 | **194,908,416** | ~29.95 | **v78 train pick** (d576 family) |
| 3 | L43×d512 mid2560 | ~194.2M | ~29.9 | |

**Conclusion:** 1-bit pack hosts a **~140M** model under the conservative estimator and **~195–197M** once C residency headroom is spent. Quality is not free: binary has no zero; need full train ladder + metrics before any ship flip.

## Binary vs best ternary

| | Params | vs freeze |
|--|--------|-----------|
| Best ternary under gate | ~80.7M | +2.4% |
| Best binary under **est** gate | ~142.3M | +80.5% |
| Best binary under **C** gate | ~197M | +150% |
| Binary − ternary densest (est) | **+61.6M** | +76% over best ternary |

## Recommendations

| Priority | Action | Why |
|----------|--------|-----|
| **1 Keep ship pack** | L23×d576 ternary+int8 | Already near ternary max at T=256; proven tools/SciQ twins |
| **2 Capability levers** | Data, tools, Engram, KD, multi-mix | Higher ROI than +2% ternary shape thrash |
| **3 If denser shape now** | Only if multi-metric win justifies retrain | Prefer modest ternary tweaks only with clear eval plan |
| **4 Binary reinvest path** | Active: v77 142M + **v78 ~195M C-headroom** | Ladder: base → multi-mix → SciQ; ship stays ternary until metrics |
| **5 Alternate density** | T=128 ternary ~99M (`train_dense_t128_max.sh`) | Already scripted; trades context, not alphabet |

## How to re-run

```bash
uv run python scripts/search_l3_capacity.py
# or
.venv/bin/python scripts/search_l3_capacity.py --out checkpoints/l3_capacity_search.json
```

Optional: `--block-size 128`, `--max-layer 48`.

## Implementation notes

- `estimate_infer_ws(..., quant="binary")` models 1-bit Linear pack (+5% pad) and optional binary embeds.
- Product serve remains **ternary**; binary is labeled capacity-search / research.
- Search grid: L∈[12,40], d∈[384,768] step 32, heads {8,12,16}, head_dim ∈ {32,40,48,64}, mid around {2…4}×d 64-aligned, **MQA only**.

## Honest knowledge note

More parameters under L3 **can** raise language/MC capacity. They do **not** replace host Engram/tools for long-tail facts. Binary’s +80% params is a **capacity ceiling estimate**, not a promise of +80% SciQ or world knowledge.

## Binary scaffold status (2026-08-05)

| Piece | Status |
|-------|--------|
| `estimate_infer_ws(quant="binary")` | done |
| STE QAT `--qat binary` | done (`ll3m/qat.py`) |
| Export `--quant binary` → **L3B2** magic | done |
| C load `L3B2` + map 1-bit Linears | done |
| C GEMV binary (scalar) | done (research; not AVX-tuned) |
| Product default | still **ternary L3T2** |
| ~142M shape train (v77) | **done** (SciQ weak after ladder) |
| **~195M C-headroom (v78)** | **training** (`run_binary_c195_ladder.sh`, `--force` for est overcount) |

Smoke: L30 binary L3B2 ≈ **19.0 MiB** weights, C total **23.0 MiB** (71.9% CCD); ternary ship C **24.3 MiB**.

## v77_binary_l30 (2026-08-07) — 142M est-gate max

| Field | Value |
|-------|--------|
| Shape | L30×d576 mid2304 MQA T=256 ≈ **142.3M** |
| Train | native binary + int8, 50k mix_train → multi-mix v15 → SciQ v16 |
| Best val CE (base) | **1.19** |
| SciQ@400 | base **17.75%**, multi-mix **19.0%** |
| C WS | weights **18.96 MiB**, total **23.0 MiB**, **7.4 MiB** under 30.4 |

## v78_binary_c195 — spend C headroom (ladder train done 2026-08-10)

| Field | Value |
|-------|--------|
| Shape | **L34×d576 mid2880** MQA T=256 ≈ **194.9M** (+37% vs v77, **2.47×** ship) |
| Why | densest-ish under C-like ≤30.4; keeps d576/h12 family |
| C-like / measured | est ~38; **C smoke total 29.56 MiB**, weights **25.52 MiB**, fill **92.4%** |
| Train | `scripts/train_dense_binary_c195.sh` + `run_binary_c195_ladder.sh` (`--force`) |
| Ladder | 60k base → +12k multi-mix v15 → +15k SciQ v16 (**done**) |
| SciQ@400 | **22.5%** (v77 sciq 19.5%; ternary p1 69.5% **unfair**) |
| Out | `checkpoints/v78_binary_c195{,_sft,_sciq}/` |
| Promote | **No**; ship stays ternary. Fairness: [BINARY_VS_TERNARY_LADDER.md](BINARY_VS_TERNARY_LADDER.md) |

**Lesson:** estimator fill ≠ C fill. Spending headroom on params is real (fits CCD). Thin binary ladder does **not** match ternary dual-ship SciQ; do not thrash short SciQ FT without recipe depth.
