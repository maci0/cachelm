# LL3M readahead design (L2 tiles + multi-session KV)

Map of **structure-known and light adaptive readahead** onto the product C path.
Not a research detour into neural address prefetchers. Goal: hide latency *inside*
the CCD-resident regime, or choose which resident bytes to keep hot when N>1.

**Constraints (non-negotiable)**

| Gate | Value |
|------|------:|
| L3 per CCD | 32 MiB |
| Product budget | ~30.4 MiB (95%) |
| Single-stream product WS | ~24.3 MiB runtime (weights ~20.3 + state) |
| Serve claim | C `c_infer/`, not Python RSS |
| Shape freeze | L23×d576 MQA, T=256, V=2048, ternary Linear + int8 embed |

If a readahead idea needs DRAM round-trips for the **hot** weight set, it is out of
scope for the product claim. Readahead here means: **software prefetch into L1/L2**,
**warm/touch policy**, and **which KV slabs stay in L3 when multi-session**.

---

## 1. Access graph (what is actually sequential)

Decode is a fixed DAG per token (product pack):

```text
embed[token]
  → for layer ℓ = 0..22:
       LN → Wq, Wk, Wv  → RoPE → attn(K/V cache[ℓ]) → Wo
         → LN → Wgate∥Wup → SwiGLU → Wdown → residual
  → LN → lm_head
```

**Block correlation is almost certain, not learned:**

| If you touch… | P(next soon) ≈ 1 |
|---------------|------------------|
| `Wq[ℓ]` | `Wk[ℓ]`, `Wv[ℓ]` (same residual `xb`) |
| `Wk/Wv[ℓ]` | RoPE + score over `k_cache[ℓ, 0:t]` |
| scores | `v_cache[ℓ, 0:t]` then `Wo[ℓ]` |
| `Wo[ℓ]` | `Wgate[ℓ]`, `Wup[ℓ]` (fused pair already) |
| `Wdown[ℓ]` | layer `ℓ+1` (or lm_head if last) |
| session S at step t | session S KV at t+1 (not session S' weights) |

Weights are **shared** across sessions; KV/state are **per stream**. That splits
readahead into two independent problems.

```text
                    ┌─────────────────────────────┐
                    │  Shared L3: weights (~20 MiB)│
                    │  Fixed order L0…L22 + head   │
                    └──────────────┬──────────────┘
           software prefetch       │
           (L2/L1 tiles)           │
                    ▼              ▼
         ┌─────────────┐   ┌─────────────┐
         │ stream 0    │   │ stream N-1  │
         │ L1 pads     │   │ L1 pads     │
         │ L2 scores   │   │ L2 scores   │
         │ KV[0:kv_cap]│   │ KV[0:kv_cap]│
         └─────────────┘   └─────────────┘
              multi: which KV slabs stay warm / which stream runs next
```

---

## 2. What we already have (baseline readahead)

### 2.1 Hierarchical warm (L3 residency)

`clm_touch_hierarchy` / `clm_warm`:

- **L1d** — dense touch of `hot_l1` (x/xb/qkv/mlp one-token).
- **L2** — stream touch of `hot_l2` (attn scores `H×Tkv`).
- **L3** — sequential 64B stream of model + state arenas.

This is **startup readahead**: force the known working set resident before timed
decode. It does not adapt online.

### 2.2 GEMV software prefetch (weight tiles)

In `c_infer/ll3m_v2.c` (ternary path):

| Hint | Use |
|------|-----|
| `CLM_PREFETCH_W` = `_MM_HINT_NTA` | next weight bytes (streaming, low pollution) |
| `CLM_PREFETCH_X` = `_MM_HINT_T0` | activations / next KV timestep |

Concrete patterns today:

- **Row inner loop:** prefetch `row + (i>>2) + 32` (~32 packed bytes ahead).
- **4-way out GEMV:** prefetch next 4 output rows' bases; also `wp + (o+4)*stride`.
- **gate∥up fuse:** dual-row prefetch on both matrices.
- **Attn:** prefetch `k_cache[u+1]` / `v_cache[u+1]` while scoring/accumulating `u`.

Product mental model in `ll3m.h`: *activation in L1, weight tiles through L2 (T1/NTA)*.

### 2.3 Multi-session capacity (+ optional RR burst)

`clm_state_create_kv(..., kv_cap)` + `multi_bench --kv-cap`:

| N streams | kv_cap | Role |
|----------:|-------:|------|
| 1–2 | 256 | full-T product / dual |
| 8 | 64–128 | balanced multi |
| 16 | ~32–48 | peak aggregate tok/s |

Layout estimate (f32 state, product shape; weights ~20.27 MiB payload):

| kv_cap | ~per-stream MiB | N=8 WS (est.) |
|-------:|----------------:|--------------:|
| 256 | ~2.23 | n/a (N=2 is the full-T multi) |
| 128 | ~1.14 | ~29.4 MiB |
| 64 | ~0.60 | ~25.1 MiB |
| 32 | ~0.33 | ~25.5 MiB (N=16) |

`hot_l2` scales with `H×kv_cap` (12 KiB @ T=256 → 1.5 KiB @ 32). Tiny vs ~1 MiB
private L2; the real multi-session pressure is **KV in L3**, not score pads.

---

## 3. Design space under the CCD gate

### Axis A — L2 / weight-tile policy (single-stream micro)

**Problem.** GEMV already streams weights. Gains are small (~1% class in
autoresearch). Remaining knobs are distance, hint level, and *which* next matrix
to pull while finishing the current one.

| Idea | Mechanism | Risk | Priority |
|------|-----------|------|----------|
| **A0. Distance sweep** | Prefetch offset `{16,32,64,128}` packed bytes; measure hard+FIFO tok/s + DRAM% | Host noise; need steady-warm | P1 micro |
| **A1. Hint level** | NTA vs T1 vs T0 for weights | T0 pollutes L1d with weights (bad for `x`) | P1 micro |
| **A2. Next-matrix cross prefetch** | While finishing `Wo[ℓ]`, prefetch first rows of `Wgate[ℓ]` / `Wup[ℓ]`; while finishing `Wdown[ℓ]`, first rows of `Wq[ℓ+1]` | Code churn; must not thrash L2 | P1 |
| **A3. Layer-ahead touch** | Soft `prefault`/`touch` of `layer[ℓ+1]` pack range during `ℓ` (only if not already resident) | Useless if fully pinned in L3; only helps soft/pagey modes | P2 / skip if mlock 100% |
| **A4. Learned tile distance** | Online EMA of “stall → increase distance” | Overfit to host load; hard to PMU-attribute | **Skip** for product |

**Decision rule for A:** keep only changes that (1) raise hard+FIFO tok/s ≥1% over
3+ runs, (2) keep runtime WS ≤ gate, (3) do not worsen GOOD DRAM% windows.

Recommended **A-stack order:** A0 → A1 → A2. Stop when autoresearch saturates.

### Axis B — KV / multi-session readahead (system)

**Problem.** Weights are shared and should stay L3-hot. Per-stream KV grows with
`N × L × kv_cap × kv_dim × 2 × 4`. Under a hard CCD gate you cannot keep full-T
KV for large N. Readahead here means: **predict which stream's KV will be used
next and keep those lines warm** (or load them before the stream is scheduled).

| Idea | Mechanism | When it helps |
|------|-----------|---------------|
| **B0. Static schedule** | Round-robin streams; each stream runs K decode steps (batch-by-stream) so KV for that stream stays in L2/L3 | Always baseline |
| **B1. Working-set aware RR** | Prefer streams whose `kv_len` is small (cheaper attn) or whose pads were touched last | Latency fairness tradeoff |
| **B2. Next-stream warm** | Before switching S→S', `touch` S' `k_cache/v_cache` only for `0:kv_len` (not full cap) | Context-switch multi on one core |
| **B3. Sliding KV window** | Already: `kv_cap < T`; treat as ring (`kv_pos0`) | Product multi N≥8 |
| **B4. Speculative dual-stream** | On two physical cores of same CCD, pin stream0/1; no DRAM if both WS fit | Dual only; CCD has 8 cores but L3 is shared |
| **B5. Session affinity map** | Host tracks (session_id → last_cpu, last_kv_bytes); pin returning session to same core | Multi-tenant chat |
| **B6. Learned P(next session)** | Markov over co-arrival of sessions | Overkill at N≤16 on one box |

**Decision rule for B:** maximize **aggregate tok/s** and **p50/p99 per-stream
latency** under `clm_inference_ws_n ≤ gate`. Prefer B0–B3 before any learning.

### Axis C — Host-side “readahead” (tools / KB) — already product-shaped

Not cache lines, but the same correlation idea:

| Access | Prefetch / place |
|--------|------------------|
| Model emits `<call>lookup …` | Host KB is already RAM-resident JSON; **no** disk readahead needed |
| Multi-call in one turn | Parallel tool fan-out (roadmap R-I) = **prefetch tool results** |
| Agent false-call basin | Gate **blocks** bad calls (measured 27/27); that is anti-readahead of wrong tools |

Do **not** pack long-tail facts into weights to avoid “readahead”; Engram-lite KB
is the deliberate placement.

---

## 4. Concrete target designs

### Design L2-1 — “Next matrix” software prefetch (single stream)

**Change (sketch in `gemv_ternary` / decode loop):**

```c
/* After last row group of Wo[ℓ], before gate/up: */
CLM_PREFETCH_W(wv->layer[li].wgate.w);      /* first cache lines */
CLM_PREFETCH_W(wv->layer[li].wup.w);
/* After Wdown[ℓ], if li+1 < L: */
CLM_PREFETCH_W(wv->layer[li + 1].wq.w);
```

**Success metric:** hard+FIFO tok/s on `v76_dense_c` pack, CCD1, steady-warm 64,
n=256 greedy; keep if ≥1% vs current keep stack; golden 16/16 still PASS.

**Non-goal:** change quant pack or WS.

### Design KV-1 — “Touch next stream” multi_bench mode

**Change:** in `multi_bench` (or a small `serve_loop` prototype):

```text
for step in 0..n_tokens:
  for i in 0..N-1:
    if i+1 < N: clm_touch_kv_prefix(&states[i+1], states[i+1].kv_len)
    clm_forward(model, &states[i]); sample...
```

Where `clm_touch_kv_prefix` only walks `k_cache/v_cache` for `L × kv_len × kv_dim`
(not full `kv_cap`).

**Success metric:** aggregate tok/s and per-stream p99 at N=8 kv_cap=64 and
N=2 kv_cap=256; WS unchanged; mlock still 100%.

**Hypothesis:** helps when streams interleave every token (worst L3 thrash).
Helps less when each stream runs a burst of K≥16 tokens (KV already warm).

### Design KV-2 — Burst scheduling (zero new touch code)

**Policy only:** each stream runs `burst=16` tokens before switch.

```text
while any active:
  pick stream i (RR)
  for k in 1..burst: decode one token on i
```

**Success metric:** same as KV-1. Often wins without prefetch because the
access graph is *session-local* for KV.

**Ship recommendation:** implement **KV-2 first** (scheduler), measure, then KV-1
only if interleave-1 is a required fairness mode.

### Design L2-2 — Prefetch distance compile-time knob

```c
#ifndef CLM_W_PREFETCH_AHEAD
#define CLM_W_PREFETCH_AHEAD 32
#endif
```

Autoresearch grid: `{0,16,32,64,128}` × hint `{NTA,T1}`. Record in
`autoresearch.jsonl`; keep only ≥1% gains.

---

## 5. What not to build (for this product)

| Idea | Why skip |
|------|----------|
| Neural address prefetcher over weight PCs | Access order is static; model is pure overhead |
| SSD/DRAM weight readahead for product serve | Violates L3-resident claim |
| Speculative MoE expert load | No MoE under freeze |
| Full-T KV for N=16 | Gate fail; use short `kv_cap` |
| Learning tile distance from production PMU without isolcpus | Confounded by host load (qemu/iframely lesson) |

---

## 6. Measurement protocol (reuse residency stack)

```bash
# Single-stream baseline (after setcap for FIFO)
taskset -c 8 ./c_infer/ll3m_c -m checkpoints/v76_dense_c/best.ll3m \
  --token-ids 12,34,56,78,90,100,110,120,130,140,150,160 -n 256 \
  --greedy --residency-hard --ccd 1 --warm 24 --steady-warm 64

# Multi
make -C c_infer multi
./c_infer/multi_bench -m checkpoints/v76_dense_c/best.ll3m \
  --streams 8 --kv-cap 64 --ccd 1 -n 128
```

Report for every readahead trial:

1. tok/s (mean of ≥3 runs)
2. runtime WS MiB / gate
3. DRAM% window if PMU on (GOOD ≤5% hard)
4. golden 16/16 if GEMV code changed

---

## 7. Implementation backlog (ordered)

| ID | Item | Effort | Depends |
|----|------|--------|---------|
| R0 | Document-only (this file) | **done** | — |
| R1 | `CLM_W_PREFETCH_AHEAD` + hint macro knobs | **done** | §10 |
| R2 | Prefetch grid (distance × hint) | **done — no keep** | `scripts/prefetch_grid.sh` |
| R3 | Next-matrix prefetch at layer boundaries | **already in tree** (gated by `CLM_NEXT_LAYER_PREFETCH`) | — |
| R4 | multi_bench `--burst K` scheduler | **done** | pthread remains product |
| R5 | `clm_touch_kv_prefix` + `--warm-next` | skip for now | R4 flat → low value |
| R6 | Optional: per-stream last-touch core affinity in a tiny serve loop | optional | R4 |
| R7 | Skip: learned Markov session predictor | skip | only if N≫16 |

**Status 2026-08-03:** R1+R2+R4 shipped and measured. No further single-stream micro
unless isolcpus-clean host; R5 not justified (burst length flat under L3-resident weights).

---

## 8. Relation to paper claims

Safe language:

- “Software prefetch streams ternary weight tiles through L2 while activations stay L1-hot.”
- “Multi-session packing under one CCD gate uses short KV (`kv_cap`) rather than DRAM offload.”
- “Host Engram-lite KB is placement of long-tail facts outside the L3 weight set.”

Avoid:

- “Learned readahead keeps weights in L3” (L3 residency is pin+size-fit, not learning).
- “Zero DRAM because of prefetch” (prefetch ≠ residency proof; PMU does).

---

## 9. Summary

| Layer | LL3M readahead form | Learn online? |
|-------|---------------------|---------------|
| Weights | Fixed layer order + GEMV NTA/T0 prefetch | No |
| L1/L2 pads | Hierarchical warm + score pad | No |
| Multi KV | `kv_cap` + burst schedule + optional next-stream touch | Optional light stats only |
| Tools/KB | Host table + gate; not weight readahead | No |

**Punchline:** under a CCD-gated dense SLM, the high-value “readahead” is
**structure-exploiting software prefetch and multi-session KV scheduling**, not a
learned block-correlation model. Learning only becomes interesting if the product
leaves full weight residency (offload / MoE / far KV tiers)—which the freeze rejects.

---

## Sources (in-repo)

- `c_infer/ll3m.h` — tier comments, `kv_cap`, touch API
- `c_infer/ll3m_v2.c` — `CLM_PREFETCH_W/X`, ternary GEMV, KV walk prefetch
- `c_infer/ll3m.c` — `clm_state_create_kv`, hierarchical touch, multi pin
- `c_infer/multi_bench.c` — N-session harness
- `docs/L3_RESIDENCY.md` — multi-session table, proof protocol
- `docs/AGENT_BATTERY_SUITE.md` — host gate as anti-false-tool “readahead”

## 10. Measured (R1/R2/R4) — 2026-08-03

### R1 knobs (shipped)

In `c_infer/ll3m_v2.c`:

| Macro | Default | Meaning |
|-------|--------:|---------|
| `CLM_W_PREFETCH_AHEAD` | 32 | packed-byte offset in ternary row loops |
| `CLM_W_ROW_PREFETCH` | 64 | bytes into current/next rows at row start |
| `CLM_W_PREFETCH_HINT` | 0 | 0=NTA, 1=T1, 2=T0 |
| `CLM_NEXT_LAYER_PREFETCH` | 1 | end-of-layer kick of next Wq/Wk/Wgate/Wup |

### R2 grid

`bash scripts/prefetch_grid.sh` (AHEAD ∈ {16,32,64} × HINT ∈ {0,1}).

| ahead | hint | tok/s |
|------:|-----:|------:|
| 16 | 0 NTA | 274.1 |
| 64 | 0 NTA | 271.9 |
| 32 | 1 T1 | 271.5 |
| 64 | 1 T1 | 271.1 |
| 32 | 0 NTA | 269.7 |
| 16 | 1 T1 | 267.5 |

**Decision: no keep.** Spread <3% with WEAK DRAM% (host load). Defaults retained.
Artifact: `checkpoints/compare/prefetch_grid/grid_20260803T015600Z.csv`.

### R4 multi `--burst`

| Mode | N=8 kv=64 agg tok/s | N=2 kv=256 agg |
|------|--------------------:|---------------:|
| pthread (product) | **1842.7** | **462.2** |
| RR burst=1 | 216.4 | — |
| RR burst=16 | 217.7 | 222.2 |
| RR burst=64 | 213.6 | — |

RR is single-core (~1/N of parallel agg). Burst length does **not** move the needle
when weights are shared and L3-resident. Keep `--burst` as a fairness / research
scheduler; do not replace pthread multi for aggregate throughput.

Golden after R1: **16/16 PASS**. Results JSON: `outputs/readahead-r1r2r4-results.json`.
