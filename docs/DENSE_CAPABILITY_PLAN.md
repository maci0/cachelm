# Dense capability plan (L3 product)

**Snapshot:** 2026-07-26  
**Goal:** max **capability per CCD byte** under freeze (L23×d576 MQA, V=2048, ternary+int8, WS ≤ ~30.4 MiB).  
**Not the goal:** encyclopedia inside ternary weights.

**Related:** [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md) · [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) · [PRODUCT_CANDIDATE.md](PRODUCT_CANDIDATE.md) · [EXPERIMENTS.md](EXPERIMENTS.md) · `checkpoints/SHIP_STATUS.json`

---

## 1. Thesis

| Layer | Holds | Location |
|-------|--------|----------|
| **Weights (L3T2)** | Language + school/common structure + tool **format/policy** | denser C / pure_v2 / future joint |
| **Host Engram-lite** | Canonical short facts | `ll3m/kb_facts.json` + `lookup` |
| **Host open world** | Long-tail / live | `wiki`, `calc`, `time` |
| **Product agent flags** | Usable tools without pure-policy pain | **assist off, gate on** (default) |

Do **not** pack more encyclopedia into ternary. Short tool/SciQ QAT thrash (d10–d14, s2) only trades poles; stop that as a density strategy.

---

## 2. Current ship (honest)

| Role | Pack | Flags | Scoreboard |
|------|------|--------|------------|
| **Product agent** | `checkpoints/v76_dense_c_pure_v2` | assist=False, **gate=True** | gate-on **101/101** (calc 30, lookup 20, no_call 24, time 5; wrong-tool rewrite; false-call **blocked 27/27**) |
| **SciQ / MC twin** | `checkpoints/v76_dense_c_sciq_trade` | N/A | SciQ **67.25** |
| **Dense language baseline** | `checkpoints/v76_dense_c` | no pure tools | SciQ **66.5**, macro ~41 |
| Pure diagnostic | same pure_v2 | assist=False, gate=False | freeze pin **90/101** (no_call 15/24); emission-aware **72/101**; user-visible **99/101** |

One pure gate-off pack with tools max + SciQ≥66.5 + no_call 24 is **not** achieved. Dual ship + product gate is the stack.

---

## 3. Workstreams (P0–P4)

### P0 — Product stack (now; no big train)

| # | Task | Status |
|---|------|--------|
| P0.1 | Document this plan + link from README / ROADMAP / SHIP_STATUS | **done** |
| P0.2 | CLI/API agent default: **assist=False, gate=True**; pure via `--pure-policy` | **done** |
| P0.3 | Expand Engram KB (entries + aliases); rebuild flat | **done** (~471→713 entries, ~1166→2167 keys) |
| P0.4 | Host gate/chitchat: greets never need tools (already When2Call) | **done** (defaults + KB assist path) |
| P0.5 | Smoke: lookup expanded keys + pure_v2 gate-on battery | **done** (99/101 gate-on 2026-07-26) |

**Success:** default `chat --mode agent` is product-safe; KB larger; docs match ship.

### P1 — Weight density (longer dense continue)

| # | Task | Notes |
|---|------|--------|
| P1.1 | Longer denser C QAT on, atomic + short edu | **done** `v76_dense_c_p1` 90k→105k |
| P1.2 | Filter capital-spam that trains greet→city prior | Still open if next data rebuild |
| P1.3 | Metric: SciQ + macro @400 | **done** SciQ **69.5** (+3), macro **41.36** (flat; BoolQ −8.3) |

**Result:** SciQ improved without macro lift (BoolQ regressed). P1 is a **SciQ peak** weight pack, not a free all-task win.  
**Artifacts:** `checkpoints/v76_dense_c_p1/std_benchmarks_limit400.json`

### P2 — One long joint stage (optional single pack)

```text
C+ (v76_dense_c_p1) → +6k steps, lr 8e-6, QAT on:
  data/p2_joint_cplus.txt
  ~43% dense/sciq plain (capital de-weighted)
  ~26% clean tool gold
  ~19% no_call hard negatives
  ~12% closed-book fact cards
```

| # | Task | Status |
|---|------|--------|
| P2.1 | Build joint mix | **done** `data/p2_joint_cplus.txt` |
| P2.2 | Train from C+ | **done** `v76_dense_c_p2` 105000→111000 |
| P2.3 | Eval SciQ400 + gate-off/on battery | **done** |

**P2 result:** gate-on **100/101** (calc 30, lookup 20, time 5); gate-off **80/101** (no_call **6/24** overtool); SciQ **65.8** (was C+ **69.5**).  
**Decision:** dual ship holds. Do **not** thrash more CE joint stages.

### P3 — Pure no_call lever (if pure AC still required)

| # | Task | Notes |
|---|------|--------|
| P3.1a | DPO 1k steps, retain 0.2 | **no ship**: no_call 24/24, calc 4/30, degeneracy |
| P3.1b | Careful DPO 150 steps, retain 0.55, β=0.05 | **research**: pure **93/101** (no_call 23/24, calc 27, lookup 16, time 5) |
| P3.1c | From p3b +100 steps, lookup-heavy retain 0.70 | **research**: pure **93/101** (no_call **24/24**, lookup **18/20**, calc 26, time 3) |
| P3.2 | KD tool-timing from Qwen | Research R-C |

**P3 lesson:** Careful DPO lifts pure total 90→93 and can reach no_call 24/24; lookup/calc/time trade under retain mix. Product gate-on still pure_v2 **99**. **Stop further DPO thrash.**  
Artifacts: `scripts/train_pref_false_call.py`, prefs JSONL, `v76_dense_c_p3b` / `p3c`.

### P4 — Host density beyond static JSON

| # | Task | Notes |
|---|------|--------|
| P4.1 | Grow KB further; structured categories | **done** **1022** entries / **3057** keys (wave2 2026-07-30) |
| P4.2 | Hashed Engram table (R-H) if JSON walk is slow | Optional |
| P4.3 | Multi-tool fan-out (R-I) | Latency, not bits |

---

## 4. Anti-patterns

| Skip | Why |
|------|-----|
| Encyclopedia in ternary | Bit budget |
| V≥32k / MoE / long ctx without residency | Breaks product |
| Weekly d/s-series short QAT | Same tools↔SciQ↔no_call basin |
| Claim gate-on as pure policy | Inflated AC |
| Capital plain-Q&A beside tool system prompts | Greets→lookup |

---

## 5. Metrics

| Metric | Role |
|--------|------|
| SciQ + macro @400 | Weight knowledge/structure |
| Gate-on battery | **Product** agent density |
| Gate-off battery | Pure stress test |
| KB n_entries / n_keys + lookup smoke | Host density |
| Infer WS ≤ 30.4 MiB | Hard gate |

---

## 6. Settled outcome (2026-07-28)

| Role | Pack / value |
|------|----------------|
| **Product agent** | `v76_dense_c_pure_v2` · assist off · gate on · **101/101** (system score; [AGENT_BATTERY_SUITE.md](AGENT_BATTERY_SUITE.md)) |
| **SciQ peak weights** | `v76_dense_c_p1` · SciQ **69.5** · macro ~41.4 |
| **Pure research** | `v76_dense_c_p3b` ~**93/101** (optional; not product default) |
| **Engram** | `ll3m/kb_facts.json` · **1022** entries / **3057** keys |
| **Paper** | v0.5+ · `docs/paper/ll3m_l3_resident.pdf` |

**Stop:** short CE joint thrash and naive long DPO. Product no_call is host gate; pure no_call is research.  
**P1 eval (2026-08-03):** gate_stats + suite doc + KB on/off + SciQ multi-seed probe — [EXPERIMENTS.md](EXPERIMENTS.md) top section; `outputs/p1-eval-report.md`.

Live status: `checkpoints/SHIP_STATUS.json` (`ship` + `dense_capability_plan`).
