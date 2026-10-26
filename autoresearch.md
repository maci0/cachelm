# Autoresearch: C decode throughput (fresh 2026-07-30)

| Field | Value |
|-------|--------|
| Started | 2026-07-30 |
| Mode | **fresh** (prior session archived under `archives/autoresearch/`) |
| Target | Speed up C inference engine |
| Metric | `tok/s` (**higher** is better) |
| Unit | tokens/second |
| Hard constraint | Engine + weights (infer WS) **fit one L3 CCD** (gate ~30.4 MiB) |
| Scope | `c_infer/` only |
| Environment | **Local** (Ryzen 9 9950X, hard+FIFO CCD1/CPU8) |
| Max iterations | **20** |
| Pack | `checkpoints/v76_dense_c/best.ll3m` (L3T2) |
| Keep rule | ≥**1%** vs best → keep; drop >0.5% or WS break → revert |
| Status | **complete** |

## Benchmark

```bash
bash autoresearch.sh
```

Hard residency, CCD1 / CPU 8, n=256, greedy, fixed token-ids.
Primary: `tok_s`. Secondary: `infer_ws_mib`, `fits_ccd`, DRAM%, FIFO.

After every rebuild: `sudo setcap cap_sys_nice+ep c_infer/ll3m_c`.

## Baseline

| Field | Value |
|-------|--------|
| Status | **done** |
| tok/s | **280.0** (279.5 / 279.8 / 280.6) |
| infer_ws_mib | **24.317** |
| fits_ccd | **YES** |
| FIFO | OK |

## Best so far

| Field | Value |
|-------|--------|
| Iteration | **15** (peak measured) |
| tok/s mean | **282.94** (5-run: 282.2–283.4) |
| tok/s median | **283.3** |
| vs baseline | **+1.05%** mean |
| Final verify | mean **281.1** / median **281.8** (5-run, host still noisy) |
| WS | **24.317 MiB**, fits_ccd=**YES** (unchanged) |
| Correctness | golden **16/16 PASS** (`scripts/compare_c_golden.py`) |

## Keep stack (in tree)

1. **unpack_ternary16** via `_mm512_insertf32x4` (no stack memcpy) — keep_marginal
2. **MQA wk+wv** fused with `gemv_ternary_pair` (stream xb once) — keep_marginal
3. **lm_head_int8** 4-way vocab rows — keep_marginal
4. **dot_f32** specialized n=48 (product head_dim) — keep_marginal
5. **MQA attention**: stream each K/V position once across heads — keep_marginal
6. **MQA first-V scale-store** (skip zero+axpy for u=0) — **keep** (+~1% peak)
7. **Next-layer weight prefetch** expanded — keep_marginal
8. **Makefile** `-fno-semantic-interposition` — keep_marginal

## Discarded (reverted)

| Iter | Hypothesis | Result |
|------|------------|--------|
| 2 | Drop full xb2 memset | −0.6%; noise-down |
| 4 | AVX silu_mul stack bounce | −1.6% |
| 5 | Dual-row gemv_ternary_pair | −1.1% |
| 8 | axpy n=48 unroll | −3.1% |
| 9 | Weight prefetch NTA→T1 | −0.7% |
| 11 | `-funroll-loops` | −2.5%; binary bloat |
| 12 | Disable LTO | flat |
| 16 | Explicit `-fomit-frame-pointer` | flat/noise |
| 17 | Dual-head score unroll | high variance |
| 19 | `in==576` fixed trip count | −2.0% |

## Host note

Load average often **~4–5** during the session (qemu + iframely + node). Early quiet runs hit **~283** tok/s; under load, same binary measures **~277–282**. Treat gains as **ranges**, not single fixed numbers. Size-fit constraint always held.

## Iterations log

See `autoresearch.jsonl` for full records.

## Policy recap

- Keep if ≥1% vs best and WS fits CCD.
- Revert if drop >0.5%, tokens diverge, or WS breaks.
- Prefer median of 5 runs when host is noisy.
