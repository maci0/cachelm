# Binary vs ternary: ladder depth and fair comparison

**Date:** 2026-08-10  
**Status:** research record (not a ship decision)  
**Related:** [L3_CAPACITY_SEARCH.md](L3_CAPACITY_SEARCH.md) · [PRODUCT_CANDIDATE.md](PRODUCT_CANDIDATE.md) · [../HANDOFF.md](../HANDOFF.md) · [SEMANTIC_COMPRESSOR.md](SEMANTIC_COMPRESSOR.md)

---

## 1. Thesis

**Ternary product twins received far more training stages, data work, and eval-driven polish than the binary research path.**  

Capacity under L3 (more params via 1-bit Linears) is a real, measured win. **SciQ / agent quality is not a fair pack-alphabet scoreboard** until binary gets a recipe depth comparable to ternary dual-ship, or we explicitly run a matched-stage ablation.

---

## 2. What “a lot more work” means (ternary)

Approximate product research stack on **ternary + int8** (L3T2), shape freeze **L23×d576 MQA**:

| Layer | Work done |
|-------|-----------|
| Architecture search | Param/gate search → freeze; peri-LN off; LayerScale 1e-4 lessons |
| Phase 1 pretrain | Long fp32 capability run on curated mix |
| Phase 2 QAT | Resume ternary Linear + int8 embed (not STE from random) |
| Dense knowledge ladder | Atomic synth, FineWeb short, MinHash, continue QAT, fact/tool SFT stages |
| Dual ship | **Agent** multi-mix (cap lineage → product gate **101/101**) + **SciQ twin** (p1 **~69.5%**) |
| Tool / agent stack | Protocol, When2Call-style mixes, constrained decode hooks, battery suite, gate rewrites |
| Host knowledge | Engram-lite KB v2, lookup aliases, tools |
| Serve proof | L3T2 export, C GEMV, residency, throughput |
| Experiment thrash | Many cap*/d*/v* continues; documented anti-patterns |

**SciQ twin reference:** `checkpoints/v76_dense_c_p1` → SciQ@400 **~69.5%**.  
**Agent ship family:** `checkpoints/v76_dense_c_pure_v2` / cap7 multi-mix lineage → battery product **101/101** (historical ship table in PRODUCT_CANDIDATE).

That is **many stages over long calendar time**, not a single base+SFT weekend.

---

## 3. What binary has received so far

### Scaffold (shared infrastructure)

- STE `--qat binary`, export **L3B2**, C load + scalar GEMV  
- Estimator `quant="binary"`; capacity search script  
- Train scripts for denser shapes  

### v77 (~142M, L30×d576 mid2304) — thin ladder

| Stage | Approx | SciQ@400 |
|-------|--------|----------|
| Base mix_train 50k native binary | done | **17.75%** |
| Multi-mix v15 SFT | done | **19.0%** |
| SciQ denser v16 | done | **19.5%** |
| Tool polish / dual-ship / dense ladder | **not** | — |
| C smoke | ~19 MiB weights, total ~**23 MiB** | fits with headroom |

### v78 (~195M, L34×d576 mid2880) — same thin ladder, more params

| Stage | Approx | Metric |
|-------|--------|--------|
| Base 60k | done | best val **1.145** |
| Multi-mix v15 +12k | done | best val **0.600** |
| SciQ denser +15k (to 86k) | done | SciQ@400 **22.5%** |
| Agent battery | in flight at handoff | TODO |
| C smoke L3B2 | done | weights **25.52 MiB**, WS **29.56 MiB**, fill **92.4%**, **fits CCD** |

Scripts: `scripts/train_dense_binary_c195.sh`, `scripts/run_binary_c195_ladder.sh`.  
Out: `checkpoints/v78_binary_c195{,_sft,_sciq}/`.

---

## 4. Scoreboard (honest)

| Artifact | Params (approx) | Pack | Recipe depth | SciQ@400 | C infer WS (order) |
|----------|-----------------|------|--------------|----------|---------------------|
| Ternary ship shape | 79M | L3T2 | **full dual-ship** | twin **~69.5** | ~24.3 MiB (79M ship) |
| v77 binary sciq | 142M | L3B2 | **thin** 3-stage | **19.5** | ~23 MiB (base shape) |
| **v78 binary sciq** | **195M** | L3B2 | **thin** 3-stage | **22.5** | **29.56 MiB** |

| Fair question | Answer from current data |
|---------------|---------------------------|
| Does binary fit more params under L3? | **Yes** (195M @ ~29.6 MiB C WS) |
| Does thin-ladder binary match ternary SciQ twin? | **No** (22.5 vs 69.5) |
| Is that proof binary alphabet is useless? | **No** (recipe depth mismatch) |
| Did more params help SciQ on thin ladder? | **Slightly** (19.5 → 22.5) |
| Promote binary to ship? | **No** until metrics + fair recipe |

---

## 5. Matched comparison design (if we continue)

Hold **eval protocol** fixed (SciQ@400 seed 42, agent battery product gate).

### Option A — Bring binary up to ternary depth (expensive)

1. Longer base or denser English/fact mix (semcomp / dense ladder style)  
2. Multi-mix agent to battery-driven early stop (not fixed short steps only)  
3. SciQ denser with **p1-class** data volume and LR schedule  
4. Optional tool SFT with CE replay  
5. Only then compare SciQ and battery to twins  

### Option B — Bring ternary recipe *down* (cheap ablation)

1. Take ternary freeze shape, run **only** the same 3 stages as v78 (base + v15 + v16 steps)  
2. Same steps/tokens if possible  
3. Compare SciQ: ternary-thin vs binary-thin  

Option B isolates alphabet + param count better. Option A asks whether binary *can* become a twin with investment.

### Option C — Stop SciQ race on binary

Keep binary as **capacity research** (fits more under CCD). Put capability investment on ternary + Engram + semcomp. Document and freeze.

---

## 6. Why thin binary SciQ stays near-random (hypotheses)

1. **Data/stage budget:** +15k SciQ continue is tiny vs p1 denser stack.  
2. **STE / no zero:** ternary has explicit zero; binary is ±1 only (harder for sparse features).  
3. **Domain tracker:** SciQ stage `best_val` stuck at SFT 0.60 (val domain mismatch); still exported `best.pt` at end from trainer logic.  
4. **Mix quality:** v16 binary SciQ mix is not the full p1 denser recipe.  
5. **Base is story-heavy** (`mix_train.bin`); knowledge density low before short SFT.

None of these are fixed by “add more layers only.”

---

## 7. Capacity side (binary win is real)

| Pack | Mechanism | Effect under 30.4 MiB gate |
|------|-----------|----------------------------|
| Ternary Linear | 2-bit + pad | ~79M product freeze near est ceiling |
| Binary Linear | 1-bit + pad | **~142M est-gate / ~195M C-calibrated** |

Measured (this host, `ll3m_c`):

- v77 L30 binary: total ~**23.0 MiB**  
- v78 L34 binary sciq export: total **29.56 MiB**, weights **25.52 MiB**  
- Ternary ship 79M: total ~**24.3 MiB**  

Estimator overstates activations; **C smoke is authority** for residency. Train used `--force` for v78.

Details: [L3_CAPACITY_SEARCH.md](L3_CAPACITY_SEARCH.md).

---

## 8. Decision record (2026-08-10)

| Decision | Rationale |
|----------|-----------|
| Ship stays **ternary** | Agent + SciQ twins proven; binary SciQ not competitive |
| v78 ladder = **research complete** on thin recipe once battery lands | Capacity OK; quality not ship-grade |
| Do not thrash more binary SciQ short FTs without recipe change | Same as cap11 lesson on ternary p1 |
| Fair next step if any | Matched thin ablation (B) or full-depth binary (A) or stop (C) |
| Semcomp | Orthogonal data densify; may help any pack; not yet on train path |

---

## 9. Artifact index

| Path | Role |
|------|------|
| `checkpoints/v78_binary_c195/` | Base |
| `checkpoints/v78_binary_c195_sft/` | Multi-mix |
| `checkpoints/v78_binary_c195_sciq/` | SciQ + eval artifacts |
| `checkpoints/v78_binary_c195_sciq/std_benchmarks_sciq400.json` | **22.5%** |
| `checkpoints/v78_binary_c195_sciq/c_smoke.log` | C WS 29.56 MiB |
| `checkpoints/v77_binary_l30_sciq/std_benchmarks_sciq400.json` | **19.5%** |
| `checkpoints/v76_dense_c_p1/` | Ternary SciQ twin ~69.5 |
| `../HANDOFF.md` | Live resume |

---

## 10. One-line summary

**Ternary: deep dual-ship recipe, strong SciQ/agent. Binary: denser L3 pack proven, thin ladder only, SciQ still near-random. Do not confuse the two.**
