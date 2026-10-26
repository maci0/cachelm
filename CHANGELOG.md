

## 2026-08-03 — docs sync (P1 eval + readahead)

- Index: `AGENT_BATTERY_SUITE`, `READAHEAD_DESIGN` in `docs/README.md` / `AGENTS.md`
- Product/ship/roadmap/experiments: gate decomp, pure dual scores, paper **v0.7.9**
- L3 + c_infer: prefetch knobs table; multi `--burst` measured notes
- IMPROVEMENTS entry for 2026-08-03



## 2026-08-03 — readahead R1/R2/R4 implemented

- R1: `CLM_W_PREFETCH_AHEAD/HINT/ROW_PREFETCH/NEXT_LAYER_PREFETCH` in `ll3m_v2.c`
- R2: `scripts/prefetch_grid.sh` — no keep under host noise (defaults retained)
- R4: `multi_bench --burst K` single-thread RR; pthread remains product multi
- Golden 16/16 PASS; results `outputs/readahead-r1r2r4-results.json`



## 2026-08-03 — readahead design note (L2 + multi KV)

- `docs/READAHEAD_DESIGN.md`: map structure-known prefetch to product C path
- Backlog R1–R4: prefetch knobs, next-matrix, multi burst schedule
- Explicit skip: neural address prefetcher / DRAM weight readahead under freeze



## 2026-08-03 — P1 gate stats + suite doc + paper v0.7.8

- `AgentTurn.gate_events` + battery `gate_stats` (block/rewrite counts)
- Product re-run: **101/101**, false-call blocked **27/27**, rewrite **2**
- Pure: emission-aware **72/101**, user-visible **99/101**, freeze pin **90/101**
- Docs: `docs/AGENT_BATTERY_SUITE.md`; tests: `tests/test_agent_gate_stats.py`
- Multi-seed helper: `scripts/run_std_benchmarks_multiseed.sh` (full grid not run)
- Paper §agent gate decomposition; still open: multi-seed n=400, ~100M peers, KB on/off



## 2026-08-02 — paper v0.7.7 P0 framing fixes

- Abstract/contributions: pure **90/101** first; product **101/101** as system×gate
- Claim→checkpoint table; size-fit + host-dependent DRAM%; name-and-operationalize novelty
- Absolute −12.8 pp MC lead; freeze pins via `paper_freeze_v072/MANIFEST.json`



## 2026-07-30 — paper v0.7.6 prose + page budget

- Tightened abstract/intro/discussion/limitations/conclusion
- Dropped redundant figures (gap relative, C+ deltas) and venue-notes appendix
- PDF **17 → 14 pages**; claims unchanged (101/101 product, 90/101 pure, 1022/3057 KB)



## 2026-07-30 — paper v0.7.5 visual polish

- Fixed eyesores: stale arch KB (713→1022/3057), broken int4 equation, pie legend, residency title/footer
- Pyramid bottom label untruncated; size-tradeoff label stagger; no overfull hbox
- Float barriers + `[H]` locks so multi-session table no longer lands in Discussion
- PDF rebuilt 17pp; figures regenerated



## 2026-07-30 — paper v0.7.3 polish

- Tighter abstract/agent prose; ship claim product **101/101**, pure **90/101**.
- Less audit meta in body; prior 99/101 kept as short variance note.
- Layout: novelty/shape/residency tables, int4 equation, artifact appendix.
- Reproducibility + appendix point at `paper_freeze_v072`.
- PDF rebuilt (tectonic).



## 2026-07-30 — paper v0.7.2 (audit fixes)

- KB counts **1022/3057** (was 895/2649) in MD + TeX.
- Product agent: current **101/101** (`battery_product_gate_on.json`); prior same-flag **99/101** documented + pinned.
- Footer fixed to v0.7.2; PDF rebuilt via tectonic.
- Pins: `checkpoints/paper_freeze_v072/` (KB + product 101/99 + pure 90 + MANIFEST).



## 2026-07-30 — autoresearch C decode (fresh, 20 iters)

- Fresh session; prior archived under `archives/autoresearch/`.
- Baseline hard+FIFO: **280.0 tok/s**, WS **24.317 MiB**, fits_ccd=YES.
- Peak keep (iter15): mean **282.9** / median **283.3** (+**~1.05%** vs baseline).
- Final verify: mean **281.1** / median **281.8** (host load ~4–5).
- Kept: insertf32x4 unpack, MQA wk+wv fuse, 4-way lm_head, hd=48 dot, MQA K/V stream, first-V scale-store, next-layer prefetch, `-fno-semantic-interposition`.
- Discarded: silu stack bounce, dual-row pair, axpy-48, T1 prefetch, funroll-loops, no-LTO, dual-head scores, in==576 special.
- Golden compare: **16/16 PASS**. L3 size-fit never broken.


## 2026-07-27 — soft ceiling + multi-stream

- Soft single-stream on CCD1: ~264 tok/s (worse than hard+FIFO ~278).
- Implemented `c_infer/multi_bench`: 2 concurrent streams share L3T2 weights.
- 2-stream WS 28.36 MiB fits 30.4 gate; **aggregate ~522–536 tok/s**.
- 3-stream WS 32.4 MiB **refused** (over gate).
- See `checkpoints/compare/l3_isolation_bench/SOFT_AND_MULTI.md`.


## 2026-07-27 — autoresearch C decode (iter 14–16)

- Softmax inv fuse + dual lm_head: flat (~281).
- znver5 march: 280.9 tok/s (keep_marginal).
- Best hard+FIFO still ~**281–282 tok/s** (+~3% vs baseline 273.3).


## 2026-07-27 — autoresearch C decode (iter 12–13)

- Online softmax: discard (−3.4%).
- True 4-row quad GEMV: discard (−2%).
- Best keep stack still ~**281–282 tok/s** hard+FIFO (+~3% vs baseline 273.3).


## 2026-07-27 — autoresearch C decode (iter 6–11)

- Kept stack to **~282 tok/s** hard+FIFO (+3.2% vs baseline 273.3): AVX attn/residual, RoPE tables, multi-row GEMV, WV2t map cache, next-layer + K/V prefetch.
- Discarded: PGO, unpack insert, fast_expf (exact tokens but −1.4% tok/s).


## 2026-07-27 — autoresearch C decode (continued)

- Iter4 PGO: **discard** (−1.7% vs best; non-PGO restored).
- Iter5 RoPE cos/sin tables: **marginal keep** ~278.4 tok/s (+0.2%).
- Best so far ~**278 tok/s** hard+FIFO (baseline 273.3).


## 2026-07-27 — autoresearch C decode

- Baseline hard+FIFO tok/s **273.3** (n=256, CCD1/CPU8, denser-C L3T2).
- Iter1 prefetch NTA→T1: no gain; discarded.
- Iter2 AVX512 attention/residual helpers on ternary decode: **277.8** tok/s (+1.6%); **kept**.
- Next: PGO and/or unpack_ternary / GEMV tile tweaks; re-setcap after rebuilds.

# CHANGELOG
