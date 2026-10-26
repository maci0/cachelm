# LL3M roadmap

**Owner:** Marcel Wysocki  
**Snapshot:** 2026-07-24  
**Product thesis:** CCD-gated co-design: full inference working set (weights + acts + KV) fits **one CPU L3 CCD** (~30.4 MiB gate); portable **C** serve (`L3T2` ternary Linears + int8 embeds); **host Engram-lite** for facts; weights hold language + tool **policy**.

**Related:** [PRODUCT_CANDIDATE.md](PRODUCT_CANDIDATE.md) · [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) · [RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md) · [HF_SURVEY_LEARN_STEAL.md](HF_SURVEY_LEARN_STEAL.md) · [ARXIV_SURVEY_LEARN_STEAL.md](ARXIV_SURVEY_LEARN_STEAL.md) · [paper/](paper/) · [../AGENTS.md](../AGENTS.md)

---

## 0. Non-negotiables (do not thrash)

| Rule | Detail |
|------|--------|
| Shape freeze | L23×d576, heads 12/1 MQA, T=256, V=2048 BPE, mid 1536, peri-LN **off**, LS **1e-4** |
| Params | **78,815,712** |
| Serve pack | Ternary Linear + int8 embed (`L3T2`); ~21 MiB weights; est. infer WS ~**30.22 / 30.40 MiB** |
| QAT | Keep **ternary + int8** through continue / SFT / RL; never drop mid-ladder |
| Facts | Host tools / KB first; **do not** pack encyclopedia into ternary weights |
| Product claim | **C** decode residency, not train RSS / Python RSS |
| Skip | MoE, MLA/DSA/Mamba at T=256, Diffusion C path, pure 1-bit (no zero), NVFP4-as-product baseline |

---

## 1. Where we are (done)

### 1.1 Product / train

| Item | Status | Artifact |
|------|--------|----------|
| Shape freeze + L3 gate | Done | `checkpoints/v75_freeze.json` |
| Base English ternary lineage | Done | `checkpoints/v76_base_en/` |
| KB v2 (entries + aliases + flat) | Done | `ll3m/kb_facts.json` (~471 / ~1164 keys) |
| Dense ladder A→D | Done | `v76_dense_{c,qa,tool}` |
| Multi-mix SFT (50/20/30 dense/fact/tool) | Done | `checkpoints/v76_dense_mix/` |
| QAT lock + best.pt promote | Done | train end + pipeline `ensure_best_pt` |
| C pack export | Done | `*.ll3m` ~20.3 MiB L3T2 |

### 1.2 Eval (settled)

| Checkpoint | MC macro (7 tasks, n=400) | Agent battery |
|------------|---------------------------|---------------|
| Prior tool path | 40.1% | (earlier) |
| **C** denser continue | **41.4%** | **100/100** |
| **D** tool/fact SFT | 40.5% (SciQ −18 pp vs C) | **100/100** |
| **MIX** multi-mix | **41.2%** (SciQ restored) | **100/100** |
| Qwen3.5-0.8B-Base **int4** | **54.2%** | n/a |
| Qwen3.5-2B / 4B / 9B int4 | **59.4 / 65.0 / 69.1%** | n/a |

Gap vs Qwen 0.8B int4: ~**−12.8 pp** (~**24%** relative). Quality compare only (portable weight-only int4; not NF4/NVFP4). Product serve remains **CPU L3**.

### 1.3 Paper

| Item | Status |
|------|--------|
| Draft (CCD-gated co-design) | **v0.7.9** Markdown + LaTeX + PDF + figures |
| Author | Marcel Wysocki (sole) |
| Path | `docs/paper/ll3m_l3_resident.{tex,pdf,md}` |

### 1.4 Default ship candidate

**Agent ship:** `checkpoints/v76_dense_c_pure_v2` **product** gate-on **101/101** (assist off; wrong-tool rewrite). Freeze: [SHIP_FREEZE.md](SHIP_FREEZE.md).  
Pure freeze pin gate-off **90/101** (calc 30/30, lookup 20/20, no_call 15/24); instrumented emission-aware pure **72/101**, user-visible **99/101**.  
Gate decomp: false-call **blocked 27/27** under product. Suite: [AGENT_BATTERY_SUITE.md](AGENT_BATTERY_SUITE.md).  
**SciQ twin:** `v76_dense_c_sciq_trade` SciQ **67.25**.  
**Density plan:** [DENSE_CAPABILITY_PLAN.md](DENSE_CAPABILITY_PLAN.md) (P0 stack → P1 long dense → P2 joint → P3 DPO).  
Live table: `checkpoints/SHIP_STATUS.json`. denser C SciQ 66.5; d14/s2 thrash stopped.  
Do **not** claim gold-inject or gate-on as pure policy.

---

## 2. North-star outcomes

1. **Ship:** Reliable short-horizon agent on CPU L3 pack (calc / lookup / no-call / time) with `--no-assist-tools` and C decode.  
2. **Measure:** Honest quality vs larger open bases at **portable weight-only int4** (not NF4/NVFP4); residency numbers for LL3M on CPU only.  
3. **Publish:** Systems paper on CCD-gated co-design + knowledge placement (not SOTA MMLU).  
4. **Improve:** Better agent reliability and dense CE under the same freeze, without growing serve WS past the gate.

---

## 3. Phased roadmap

### Phase P0 — Close measurements (now)

| # | Task | Why | Success | Status |
|---|------|-----|---------|--------|
| P0.1 | **Qwen3.5 family weight-only int4** MC (0.8B→9B, limit 400) | Size ladder vs LL3M C/MIX | JSON + macro/per-task table | **Done** (int4 only; NF4 JSONs deleted) |
| P0.2 | Optional: Qwen3.5-**27B** weight-int4 | Upper curve only | Same protocol | Pending |
| P0.3 | Aggregate `ll3m_vs_qwen35_family_int4.json` | Paper/table one-shot | File in `checkpoints/compare/` | **Done** (int4-only filter) |
| P0.4 | Pull macros into paper §Results | Publish readiness | Updated tex/pdf | Pending |
| P0.5 | **int4 only** for quality compare | No NF4/NVFP4 baselines | Framing only | Policy |

**Scoring note:** Qwen MC may run on CUDA for throughput; accuracy is quality-only (`--weight-int4` per-row absmax). Runtime/residency claims stay **CPU + C**.

Scripts: `scripts/run_std_benchmarks_hf.py --weight-int4`, `scripts/run_qwen35_family_int4.sh`, `scripts/aggregate_qwen_family_vs_ll3m.py`.

---

### Phase P1 — Ship the agent (product)

| # | Task | Why | Success |
|---|------|-----|---------|
| P1.1 | Export MIX → `best.ll3m`; C smoke + `compare_c_golden` | Train–serve lock | Goldens pass under ternary | **Done** (MIX golden 16/16; aarch64 C rebuild) |
| P1.2 | Agent battery **`--no-assist`** (gate+assist off) on C / D / MIX | Real policy, not host gold | Report pass rate + failures | **C 100/101 · D 37/101 · MIX running** |
| P1.3 | Fix residual agent footguns (format, spam, recovery) | Ship reliability | Documented battery green path |
| P1.4 | Soft residency proof on product host | Paper + claim | Soft protocol log; optional hard later |
| P1.5 | Chat UX defaults: MIX pack, agent mode, gate_tools on | Product path | One-liner in README |

**Out:** “v76 product” = MIX weights + C pack + host KB + documented eval.

---

### Phase P2 — Short-horizon agent RL (GLM / Laguna-inspired, SLM-cheap)

Inspired by **GLM-5.2** (critic-PPO for *long* agents; GRPO still fine for short groups) and **Laguna XS/S 2.1** (SFT then RL on hard envs, multi-harness, persistence, verifiers). **Do not** copy MoE / 1M ctx / full slime.

| # | Task | Inspiration | Success |
|---|------|-------------|---------|
| P2.1 | Verifiable reward suite: calc exact, lookup hit, no-call, format | Laguna envs; GLM anti-hack | **Done** `ll3m/rewards.py` + tests |
| P2.2 | **GRPO-lite / RLOO** (N=4–8), short episodes, **QAT on** | DeepSeek GRPO; not full PPO yet | Lift no-assist battery without SciQ collapse |
| P2.3 | Dummy/error tool results + recovery in SFT | GLM-5.2 anti-hack | **Done** `gen_tool_sft` v2 + self-distill harvest |
| P2.4 | Multi-format tool prompts (same tools, slight schema noise) | Laguna multi-harness | Less harness overfit |
| P2.5 | Compaction SFT: answer from last user + last tool only | GLM compaction; our T=256 | Works under clipped history |
| P2.6 | Defer full **critic PPO** | GLM-5.2 long-horizon only | Only if P2.2 saturates on multi-hop |

**Out:** RL-polished MIX-class checkpoint under same freeze.

---

### Phase P3 — Capability under the gate (quality)

| # | Task | Inspiration | Success |
|---|------|-------------|---------|
| P3.1 | Longer multi-mix / denser continue (+steps, low LR) | Phi overtrain; our ladder | MC + samples vs MIX |
| P3.2 | Teacher **atomic** science/geo pack (short cards only) | Phi textbooks | ARC-E / SciQ lift |
| P3.3 | **KD** into L23×d576 from Qwen 0.8B–1.7B class | Ministral cascade; R-C | Val CE + MC vs CE-only |
| P3.4 | Grow host KB (canonical + aliases), not weights | Engram; GLM retrieval-lift | Lookup hit rate; agent facts |
| P3.5 | Optional tiny draft / speculative path for C | Laguna DFlash idea | tok/s if CPU-bound |

**Out:** Higher MC and denser agent skill **without** breaking 30.4 MiB gate.

---

### Phase P4 — Quant / C systems

| # | Task | Notes |
|---|------|--------|
| P4.1 | Serve-grid quality: fp32 export vs QAT vs native | `compare_fp32_vs_qat.py` |
| P4.2 | First-class `BitLinear` STE module | Replace monkey-patch |
| P4.3 | Train–serve absmean / pack lock audit | bitnet.cpp lesson |
| P4.4 | Optional TL/ternary GEMV | Only if decode-bound |
| P4.5 | Optional W1.58A8 | After weight QAT solid |
| P4.6 | Hard residency (size-fit + mlock + FIFO; **no isolcpus** on product host) | Measured; DRAM WEAK accepted |

---

### Phase P5 — Publish

| # | Task | Notes |
|---|------|--------|
| P5.1 | Insert Qwen family int4 table (0.8B→9B) | After P0 |
| P5.2 | Optional: no-assist battery + residency soft numbers | After P1 |
| P5.3 | Expand BibTeX; affiliation optional | Author set |
| P5.4 | arXiv or workshop (MLSys / on-device / efficient-ML) | Claim: CCD-gated co-design |

Primary claim: **constraint-driven co-design under one L3 CCD**, not SOTA MMLU.

---

## 4. Research inputs (mapped)

### 4.1 Already absorbed

| Source | Lesson in product |
|--------|-------------------|
| BitNet / ternary | STE + int8 emb pack; continuous QAT |
| Engram / GLM retrieval-lift | Host KB + tools ≫ weight facts |
| Phi / FineWeb-Edu | Dense cards + edu filter + near-dedup |
| When2Call / BFCL | no-call balance; tool SFT |
| Multi-mix experiment | Tool-only SFT hurts SciQ; replay fixes it |

### 4.2 GLM-5.2 (use carefully)

| Take | Leave |
|------|--------|
| Short-horizon GRPO still fine for SLM tools | Full critic PPO at our size unless multi-hop |
| Anti-hack: dummy tool results, continue | slime-scale async mega-infra |
| Compaction-aware trajectories | DSA / IndexShare / 1M ctx |
| Verifiers on tool outcomes | Merging 10+ expert models |

### 4.3 Laguna XS 2.1 + S 2.1 (use carefully)

| Take | Leave |
|------|--------|
| Same pretrain data; **post-train** is the jump XS→S | MoE 33B / 118B |
| SFT then RL on **hard, verifiable** agent envs | 256K–1M context |
| Multi-harness rollouts | Matching Terminal-Bench 70% |
| Persistence / verify / don’t early-exit | NVFP4 as quality baseline |
| Thinking as test-time compute | Copy multi-hour think traces into T=256 |
| XS: local product mindset | Expect S-class long-horizon coding at 79M |

**Thinking on LL3M:** no 100k-token monologues. Prefer short plan tokens **or** host-side verify loops (extra tool rounds).

---

## 5. Priority queue (next actions)

Ordered for default execution:

1. **P0 done** — Qwen family weight-only int4 (0.8B→9B); NF4 artifacts deleted.  
2. **P1.1–P1.2** — C golden + no-assist battery on MIX (in flight on spark1).  
3. **P2.1–P2.2** — Verifiable rewards + GRPO-lite under QAT.  
4. **P3.3 or P3.1** — KD or longer multi-mix (pick by GPU budget).  
5. **P5** — Paper refresh with int4 family table.

---

## 6. Ops map

| Host | Role |
|------|------|
| Local (ROCm 9950X class) | Product serve, residency, C build, paper |
| spark1 CUDA | Long GPU train / HF MC (Qwen family, dense ladder) |
| spark2 | Prior base-English / native trains (historical) |

| Screen / log (spark1) | Purpose |
|-----------------------|---------|
| `ll3m-dense` | Dense ladder (done) |
| `ll3m-mix` | Multi-mix + batteries (done) |
| `ll3m-qwen-int4` | Qwen family weight-only int4 (done) |
| `ll3m-ship-b` | Export + golden + no-assist batteries (active) |
| `checkpoints/dense_pipeline/*.log` | Pipeline logs |

---

## 7. Success scoreboard

| Metric | Now | Near target |
|--------|-----|-------------|
| MC macro (C/MIX) | ~41.2–41.4% | ≥43% without WS growth |
| vs Qwen 0.8B int4 | −12.8 pp | Document only; not parity goal |
| vs Qwen 9B int4 | −27.7 pp | Document size gap honestly |
| Agent battery (assist) | 100/100 C/D/MIX | Keep for gated UX |
| Agent battery (no-assist) | **C 100/101** (ship); MIX 70/101; D 37/101 | Core cats maxed on C |
| L3 pack | ~20.3 MiB L3T2 | Still under gate |
| Paper | Draft v0.4 | arXiv-ready with 9B table |

---

## 8. Explicit anti-roadmap

- Grow model past CCD gate without a new freeze write-up.  
- Tool-only SFT without dense CE replay.  
- Pure closed-book knowledge race against 0.5B–9B.  
- NVFP4 / GPU-only formats as the main quality story.  
- MoE / long-context / MLA as default product path at T=256.  
- Full GLM-scale critic PPO before short GRPO is tried.

---

## 9. Doc maintenance

| When | Update |
|------|--------|
| Phase gate completes | This file §1 status + §7 scoreboard |
| New freeze | `v75_freeze.json`, PRODUCT_CANDIDATE, TARGET_75M |
| New train recipe | DENSE_KNOWLEDGE_LADDER, AGENTS train recipes |
| Paper numbers | `docs/paper/` |

---

*End of roadmap snapshot 2026-07-24*
