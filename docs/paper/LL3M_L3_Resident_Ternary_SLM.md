# LL3M: CCD-Gated Co-Design of a Ternary Language Model for One L3 Slice

**Author:** Marcel Wysocki  

**Draft for submission** (not yet peer-reviewed)  
**Version:** 0.7.9 · **Date:** 2026-08-03  
**LaTeX:** [`ll3m_l3_resident.tex`](ll3m_l3_resident.tex) · **Bib:** [`ll3m_l3_resident.bib`](ll3m_l3_resident.bib) · **PDF:** [`ll3m_l3_resident.pdf`](ll3m_l3_resident.pdf) · **Build:** [`README.md`](README.md)

**Keywords:** CCD-gated co-design; L3-resident inference; ternary QAT; knowledge placement; tool-augmented SLMs  

---

## Abstract

Most "small" LMs still target phones or discrete GPUs with multi-hundred-megabyte payloads. We ask a different question: *what if the product unit is one CPU last-level cache (L3) CCD?* **LL3M** co-designs architecture, ternary/int8 quantization, portable C packing, curriculum, and host memory so the *entire* inference working set (weights + activations + KV) **size-fits** a single CCD gate (~30.4 MiB on a 32 MiB slice).

The freeze is ~79M parameters (L23×d576 MQA, T=256, V=2048), served as a **20.3 MiB** L3T2 pack with continuous QAT matching C. **Knowledge placement** is first-class: ternary weights hold language and tool policy; facts live in a host Engram-lite KB and tools. On seven zero-shot MC tasks (n=400, denser-continue pack), macro is **41.4%** vs **54.2%** for Qwen3.5-0.8B-Base under portable weight-only int4 (**−12.8 pp** absolute; ~9.5× more Qwen params; not iso-tokenizer/data). Tool-only SFT regresses SciQ by 18 pp; multi-mix SFT restores SciQ (macro 41.2%). On a **closed** 101-case agent suite (not BFCL), the tool-SFT ship pack scores **90/101 pure** (assist and gate off; no_call 15/24) and **101/101 product** only with assist off and a When2Call-style **host gate on**—product quality is model emission × host policy, not model-only competence. Artifact pins: `checkpoints/paper_freeze_v072/` (SHA256 in `MANIFEST.json`).

On the reference host, C decode **size-fits** one CCD (runtime WS **24.3 MiB**; design **30.2 MiB**) with 100% mlock; hard+FIFO holds ~**280–282 tok/s**. DRAM share of L2 fills is often **WEAK** without isolation; **GOOD** (≤5%) appears only under hard+FIFO / quiet-core (preferably isolcpus) paths—not a portable zero-DRAM guarantee. Multi-session packing under the same size gate reaches N=2 at full T (~550 tok/s aggregate) and N=8 with **short KV** (~2100–2200 aggregate). We **name and operationalize** **CCD-gated co-design**: a hard L3 size-fit constraint forcing joint shape/quant/serve/data/host choices, with train–serve lock, honest PMU, and dual product/pure agent metrics.


---

## 1. Introduction

### 1.1 Motivation

Large language models deliver strong closed-book knowledge but require multi-gigabyte (often multi-ten-gigabyte) payloads. Many “small” models marketed for phones and laptops still use **0.5–4B** dense weights in fp16/bf16 or 4-bit, far beyond a single CPU L3 slice. In contrast, modern CPUs expose **tens of mebibytes** of L3 per complex. If the **entire** decode working set can remain resident in one L3 CCD, steady-state decode can avoid DRAM round-trips for weights, activations, and KV: a systems-level advantage distinct from FLOP counts alone.

### 1.2 Research questions

1. What architecture and quantization grid maximize parameters under a fixed L3 inference gate?  
2. Can **native ternary QAT** (BitNet-style absmean STE) produce a serve pack that matches the C product format without a separate post-hoc quant catastrophe?  
3. How should knowledge be allocated between **weights**, **train tokens**, and **host memory** at this scale?  
4. How does the resulting system compare to a much larger industry base (Qwen3.5-0.8B) on standard MC benchmarks, and what agent trade-offs appear under honest policy flags?

### 1.3 Contributions

1. **L3-first product freeze:** L23×d576 MQA, \(T{=}256\), \(V{=}2048\) BPE, **78,815,712** parameters, ternary Linears + int8 embeds, design-time infer WS **~30.22 MiB / 30.40 MiB** gate (**~99.4%** fill). C runtime payload WS on the product pack is **~24.3 MiB** (weights file **20.3 MiB**).  
2. **Portable C serve path** with residency controls (pin, warm, CCD bind, optional hard isolation), product claim scoped to **C**, not Python train RSS.  
3. **Knowledge split:** host Engram-lite KB (v2: **1022** entries / **3057** keys) + tools; weights hold language and tool **policy**.  
4. **Dense ladder + dual agent metrics:** atomic synthetic + FineWeb-Edu (short-capped) + MinHash near-dedup; continue-pretrain and multi-objective SFT with QAT never dropped.  
5. **Measured comparison** to Qwen3.5 family weight-only int4 on seven standard tasks, longer dense continue (SciQ specialization), closed agent batteries under **product gate / pure policy**, and **on-host residency measurements** (size-fit PASS; PMU DRAM% WEAK without full isolation).

### 1.4 Non-goals

We do **not** claim closed-book parity with 0.5B–8B models, trillion-token overtraining budgets, that training working sets fit L3, or that PMU DRAM share is already ~0% on an unisolated desktop. We do **not** recommend MoE, multi-million-token context, or pure 1-bit (no-zero) weights for this product. We do **not** claim SOTA on BFCL or public agent leaderboards; our agent suite is a closed product battery.

---

## 2. Related work

### 2.1 Extremely small LMs and data quality

Phi (“Textbooks Are All You Need”), TinyStories, and SmolLM emphasize **educational filtering** and synthetic textbooks over raw web scale alone. MobileLLM studies sub-billion on-device shapes. LL3M adopts the density intuition at a much smaller parameter budget and with an explicit **cache gate** that forbids large vocabs and long contexts those lines often retain.

### 2.2 Ternary / 1.58-bit training and serve

BitNet and BitNet b1.58 train with ternary \(\{-1,0,+1\}\) weights via STE and absmean scaling. bitnet.cpp stresses **train–serve lock** of packing (I2_S lesson). LL3M uses ternary Linears + int8 embeds (not pure 1-bit), QAT throughout dense continue and SFT, and a C **L3T2** pack under a fixed WS ceiling.

### 2.3 External memory and tools

DeepSeek **Engram** formalizes conditional O(1) memory outside the residual stream. Tool-calling literature (Gorilla/BFCL, When2Call, SLM function-calling finetunes, LaCy) shows small models need **narrow schemas**, gold SFT, and **no-call** negatives. LL3M implements a closed registry (`calc`, `lookup`, `time`, `wiki`, `note`) with a short `<call>…</call>` protocol and an optional host When2Call-style gate.

### 2.4 Cache-aware and on-device inference

Prior systems (llama.cpp, MLC-LLM, PowerInfer, mobile NPU stacks) optimize kernels, paging, and quantization for average-case locality or heterogeneous memory. Our contribution is a **hard product constraint**: one L3 CCD for the full infer WS at design time, plus residency hooks and measured PMU on the C path, rather than average-case caching alone.

---

## 3. System design

### 3.1 Architecture

| Component | Choice |
|-----------|--------|
| Block | Pre-norm Transformer (v2): RMSNorm, RoPE, SwiGLU, QK-Norm, no projection bias |
| Shape | **23 layers**, **d_model=576**, **12 Q heads / 1 KV head (MQA)**, mid **1536**, head_dim **48** |
| Context | \(T{=}256\) |
| Vocab | Domain BPE **2048** (embed table must not dominate L3) |
| Stability | LayerScale init **1e-4**; peri-LN **off** (blocked early learning on this shape) |
| Train extras | PaLM-style z-loss, shared-head multi-token prediction (train-only) |
| Parameters | **78,815,712** |

MQA is preferred at short \(T\) under the gate: KV is cheap, so width/depth can absorb more parameters.

### 3.2 Quantization and serve pack

- **Linears:** ternary STE absmean during train; packed 2-bit with scale in C (**L3T2**).  
- **Embeddings:** int8 QAT (ternary embeds cost quality for tiny L3 win).  
- **Export:** `export_weights.py --quant ternary` → `.ll3m` (**20.3 MiB** on disk for denser-C).  
- **Python train** may use full DRAM/GPU; **product claim** is C decode WS.  
- **Same-grid note (v75 lineage):** post-hoc forcing ternary+int8 on an fp32 checkpoint raised val CE from **1.14 → 2.27**; native ternary+int8 QAT on the same shape sits at val CE **~1.26** on the serve grid (`checkpoints/v75_ternary/quality_compare.json`). This supports continuous QAT over naive post-hoc pack.

Portable weight-only int4 baseline (Qwen) uses per-row absmax fake-quant of Linear weights (not NF4/GPTQ/NVFP4):

\[
\hat w_{i,j} = s_i \cdot \mathrm{clip}\!\left(\mathrm{round}(w_{i,j}/s_i),\,-8,\,7\right),\quad
s_i = \max_j |w_{i,j}| / 7.
\]

### 3.3 L3 residency

On the reference host (AMD Ryzen-class **2×32 MiB** L3 CCD, product budget **95% × 32 MiB ≈ 30.4 MiB**):

\[
\mathrm{WS}_{\mathrm{design}} \approx W_{\mathrm{weights}} + W_{\mathrm{acts}} + W_{\mathrm{KV}}.
\]

| Quantity | Value | Role |
|----------|------:|------|
| Design estimator (`estimate_infer_ws`, `quant=ternary`) | **30.22 MiB** | Shape freeze / refuse-if-over-gate |
| Product gate (0.95×32 MiB) | **30.40 MiB** | Admissibility |
| C runtime payload WS | **24.32 MiB** | weights **payload** 20.27 + state/KV arena 4.00 + pads |
| L3T2 weights **payload** (C arena) | **20.27 MiB** | Runtime weight bytes after load |
| L3T2 **on-disk** `.ll3m` | **20.32 MiB** | Filesystem size (header/padding; denser-C `best.ll3m`) |
| CCD L3 | **32.0 MiB** | Hardware slice |

Soft/hard residency modes include CCD bind, `mlock`, hierarchical warm, optional single-core isolcpus / SCHED_FIFO / hugepages. Training tensors are **not** required to fit L3.

### 3.4 Host knowledge (Engram-lite)

Closed facts live in `kb_facts.json` (v2: canonical ids, aliases, flat index; **1022** entries / **3057** keys). `lookup` uses **word-bounded** matching (avoid `pi` ⊂ `pizza`). Open-world facts use capped Wikipedia summaries. Arithmetic uses host `calc`. The model learns **when** to emit tool calls; the host executes them. Host table growth is free density relative to ternary weight bits. KB on/off (lookup n=20, pure_v2): full KB 20/20; empty KB 0/20 under product and pure. Ship pins for KB + product/pure batteries: `checkpoints/paper_freeze_v072/`.

```
User → [decode in L3] → plain answer | <call>tool args</call>
                              ↓
                    host execute → <result>…</result>
                              ↓
                    [decode continues] → final answer
```

---

## 4. Training curriculum

### 4.1 Stage 0: shape search and freeze

Exhaustive max-parameter search under the gate with practical head sets yields the L23×d576 MQA freeze (documented in product notes). Peri-LN ablations on this width failed early (val CE stuck ~5.9); LayerScale alone progressed.

### 4.2 Stage 1: base English (native ternary)

Shift from TinyStories-primary to **base English**: FineWeb-Edu-style educational web + dense synthetic cards, native `--qat ternary --embed-qat int8` from step 0 or long ternary continue. Checkpoint lineage used here: `v76_base_en` (step ~60k, already ternary+int8).

### 4.3 Stage 1b: dense knowledge ladder

| Exp | Action |
|-----|--------|
| A | Expand host KB v2; validate load |
| B | 700M synthetic (atomic-biased) + 500M FineWeb (doc cap 1200); exact + MinHash near-dedup (Jaccard 0.85); short-doc bias; BPE \(V{=}2048\) → **~161M** train tokens |
| C | Continue ternary QAT **+30k** steps on denser mix → `v76_dense_c` |
| C+ (P1) | Further **+15k** dense continue under QAT → `v76_dense_c_p1` |
| D | Fact Q&A SFT then tool SFT (QAT on) → `v76_dense_tool` |
| Mix | Multi-objective SFT from C: **~50% dense / 20% fact / 30% tool**, +5k steps → `v76_dense_mix` |
| Ship tools | Tool-focused SFT lineage → `v76_dense_c_pure_v2` (agent ship weights) |

**QAT lock:** every stage asserts `qat=ternary` and `embed_qat=int8` on checkpoints. End-of-run `best.pt` is written even if val never beats a stale resume bar (domain-shift fix).

**Train hosts:** dense ladder and Qwen int4 benches primarily on remote CUDA host `spark1`; product C residency measured on the local Ryzen L3 reference machine. Exact wall-clock GPU-hours are not yet fully logged for all stages (see Limitations).

### 4.4 Failure mode observed

Continue-pretrain on denser data with **unchanged QAT grid** retained a base `best_val` bar; denser-mix val never beat it mid-run, so only `last.pt` was written until recovery/promotion. Final denser val recovered (ppl ~13.8 at end of C). **P1** again never beat resume `best_val` (2.517) yet **SciQ rose 66.5→69.5**: multi-domain val CE is a **false friend** for early stop; report end-of-run weights + MC, not only best_val.

---

## 5. Evaluation protocol

### 5.1 Multiple-choice suite

Zero-shot **continuation log-likelihood** ranking, seed 42, limit **400** examples per task (COPA uses full set **n=100** when smaller):

HellaSwag, ARC-Easy, WinoGrande, BoolQ, OpenBookQA, SciQ, COPA.

**Macro accuracy** = unweighted mean of task accuracies.  
**Caveat:** domain BPE \(V{=}2048\) vs Qwen’s large vocab makes LL ranking a harsh, not iso-tokenizer, comparison. OpenBookQA below chance for LL3M may partly reflect format/tokenization, not only knowledge absence. Single seed; no error bars.

### 5.2 Baseline

**Qwen3.5 Base family** under **portable weight-only int4** (per-row absmax; §3.2), same tasks/limits/seed. Primary: 0.8B-Base at **54.21%**; ladder also 2B/4B/9B. Industry bases, **not** L3-resident packs. Canonical artifacts: `checkpoints/compare/Qwen__Qwen3.5-*-Base_int4_limit400.json` and aggregate `ll3m_vs_qwen35_family_int4.json`.

**Missing baselines (explicit):** same-shape fp32 student on the dense ladder MC suite; public ~50–100M SLMs under our harness; bitnet.cpp latency peer; NF4 Qwen. A v75-lineage val-CE grid (§3.2) partially addresses post-hoc vs QAT.

### 5.3 Agent battery (evaluation card)

Closed suite (**101** cases): no-call chat, calc, lookup, time, language openers, adversarial, tokenizer. **Not** BFCL / public When2Call leaderboard.

| Policy | assist (gold inject) | gate (When2Call-style) | Role |
|--------|----------------------|------------------------|------|
| **Product** | off | **on** | Ship claim = model × host gate |
| **Pure** | off | **off** | Model-only stress |
| Assist / gold | on | either | Bootstrap ablation only |

Decode is C L3T2 unless noted; host executes tools after `<call>…</call>`. Greedy pure-policy sampling unless noted.

**Canonical pure / product files (use these):**

| Lineage | Product (gate on) | Pure (gate off) |
|---------|-------------------|-----------------|
| denser C | (language baseline) | `checkpoints/v76_dense_c/battery_true_pure_greedy.json` **42/101** |
| tool ship pure_v2 | product **101/101** (pin `paper_freeze_v072/battery_product_gate_on_101.json`) | pure **90/101** (pin `paper_freeze_v072/battery_true_pure_greedy_90.json`) |
| D tool/fact | — | `checkpoints/v76_dense_tool/battery_D_no_assist.json` **37/101** |
| MIX multi-mix | — | `checkpoints/v76_dense_mix/battery_MIX_no_assist.json` **70/101** |

Bare basenames collide across dirs (e.g. denser-C pure **42/101** vs pure_v2 pure **90/101**); always use the full path.

A conflicting denser-C file that reported 100/101 with full tool passes is **quarantined** (`checkpoints/v76_dense_c/_quarantine/`); do not cite it.

### 5.4 Residency measurement card

```text
./c_infer/ll3m_c -m checkpoints/v76_dense_c/best.ll3m \
  --token-ids … -n 64|128 --greedy \
  --l3-resident | --residency-hard --warm 16|24 --steady-warm 32|64 --ccd 0
```

Logs: `checkpoints/compare/residency_{soft,hard}_v76_dense_c.txt`.

---

## 6. Results

### 6.1 Closed-book macro vs Qwen

| Model | Params (approx.) | Serve footprint (approx.) | Macro acc. |
|-------|------------------|---------------------------|------------|
| Qwen3.5-0.8B-Base int4 | ~0.75B | multi-GB class | **54.21%** |
| Qwen3.5-2B / 4B / 9B int4 | ~1.9B / 4.2B / 9.0B | multi-GB class | **59.4 / 65.0 / 69.1%** |
| LL3M prior tool path | ~79M ternary | **20.3 MiB** L3T2 | 40.11% |
| LL3M denser continue (**C**) | ~79M ternary | **20.3 MiB** L3T2 | **41.39%** |
| LL3M dense continue **+15k** (**C+** / P1) | ~79M ternary | **20.3 MiB** L3T2 | **41.36%** |
| LL3M tool/fact SFT (**D**) | ~79M ternary | **20.3 MiB** L3T2 | 40.46% |
| LL3M multi-mix SFT (**MIX**) | ~79M ternary | **20.3 MiB** L3T2 | **41.18%** |

**Gap (C vs Qwen 0.8B int4):** −**12.82** percentage points absolute; relative deficit \(1 - 41.39/54.21 ≈ \) **23.6%**.  
Vs 9B int4: −**27.7** pp. LL3M remains a **~20.3 MiB** L3T2 pack under a CCD gate the Qwen bases do not target. **C+ does not lift macro** despite SciQ gains (next table).

### 6.2 Per-task (portable int4 Qwen column)

Qwen numbers from `Qwen__Qwen3.5-0.8B-Base_int4_limit400.json` (macro 54.21%). COPA n=100; other tasks n=400.

| Task | C | **C+** | D | MIX | Qwen int4 0.8B | Random (approx.) |
|------|--:|-------:|--:|----:|---------------:|------------------:|
| HellaSwag | 28.0 | 28.5 | 26.5 | 29.0 | 35.5 | 25 |
| ARC-Easy | 27.3 | **29.8** | 31.0 | 27.8 | 59.0 | 25 |
| WinoGrande | 47.5 | 48.0 | 51.0 | 50.0 | 49.8 | 50 |
| BoolQ | **56.0** | 47.8 | 54.3 | 46.5 | 63.0 | 50 |
| OpenBookQA | 13.5 | 13.0 | 16.0 | 12.5 | 21.0 | 25 |
| SciQ | 66.5 | **69.5** | 48.5 | 66.5 | 84.3 | 25 |
| COPA | 51.0 | 53.0 | 56.0 | 56.0 | 67.0 | 50 |

**Takeaway (SFT):** Tool-only SFT (**D**) hurts SciQ (−18 pp vs C) while helping some two-way tasks. **Multi-mix SFT restores SciQ** and keeps macro near C: CE replay + tools beats tool-only fine-tuning at this scale.

**Takeaway (C→C+):** Further dense continue under QAT **specializes** the pack: SciQ **+3.0** and ARC-Easy **+2.5**, but BoolQ **−8.25**, macro **unchanged** (~41.4%). SciQ is not a proxy for seven-task macro. Val CE never beat the resume bar on C+, yet end-of-run weights improved SciQ.

### 6.3 Agent battery (honest flags)

Primary ship pack: tool SFT lineage (**pure_v2**), C L3T2 decode, greedy sampling.

| Policy / lineage | assist | gate | Total | calc | lookup | no_call | Role |
|------------------|:------:|:----:|------:|-----:|-------:|--------:|------|
| **Product pure_v2** | off | **on** | **101/101** | 30/30 | 20/20 | 24/24 | Ship (model×gate) |
| **Pure pure_v2** | off | **off** | **90/101** | 30/30 | 20/20 | 15/24 | Model stress |
| denser C pure | off | off | 42/101 | 0/30 | 0/20 | 24/24 | Language baseline |
| D pure | off | off | **37/101** | 3/30 | 7/20 | 8/24 | Tool-only SFT |
| MIX pure | off | off | **70/101** | 24/30 | 19/20 | 2/24 | Multi-mix SFT |

Product pure_v2 is **101/101** (time 5/5) on the ship pin. A prior product-flag re-run scored **99/101** (2/5 `time` fails, not no_call); both are under `checkpoints/paper_freeze_v072/`. Retired claims: pure 100/101 or denser-C 100/100 without flag discipline (gold inject / gate conflation / quarantined artifact). Product quality is **(model emission) × (When2Call gate)**. Instrumented re-run: model false-called on 27 no-tool cases; host **blocked 27/27** under product gate (see `docs/AGENT_BATTERY_SUITE.md`) for no-call; pure gate-off remains imperfect. MIX improves tools over D but collapses pure no_call (2/24) without the host gate.

Train–serve greedy match on the tool ship pack: **16/16** tokens (C golden).

### 6.4 Policy–knowledge Pareto (short QAT)

Short ternary QAT from either pole (tools-max or SciQ-max) does **not** yield one pack that jointly maximizes tool accuracy, SciQ, and pure no-call. Denominators shown. SciQ for d12/s2 from SciQ-only runs (same seed/limit); pure_v2/sciq_trade/C from full 7-task JSON where available.

| Lineage (gate-off pure) | calc | lookup | SciQ % | Pure no_call | Note |
|-------------------------|-----:|-------:|-------:|-------------:|------|
| denser C | 0/30 | 0/20 | 66.5 | 24/24 | No tool policy |
| pure_v2 | **30/30** | **20/20** | 57.75 | 15/24 | Agent ship weights |
| sciq_trade | 14/30 | 14/20 | **67.25** | 9/24 | SciQ SFT twin |
| C+ (P1) | (no tool SFT) | (n/a) | **69.5** | n/a | SciQ peak weights |
| tool flood (d12) | 30/30 | 20/20 | 60.3 | **7/24** | Overtool basin |
| sciq→tools (s2) | 27/30 | 19/20 | 65.8 | **8/24** | Same basin |
| MIX | 24/30 | 19/20 | 66.5 | **2/24** | CE replay helps SciQ, not no_call |

**Implication:** dual ship (agent pack + SciQ/weight twin) and/or **product gate** are design outcomes under the CCD budget, not temporary engineering debt. Recovering pure no_call after tool-mass collapse is hard with CE alone.

**Preference follow-up (P3):** naive DPO on false_call pairs (1k steps) reaches pure no_call 24/24 but destroys tools and fluency. Careful DPO (150 steps, ~55% tool CE retain, low β) raises pure total **90→93/101** (no_call 23/24) while product gate-on remains strongest on pure_v2 (**101/101**). Product no_call is therefore primarily a **host gate** result; pure no_call is a research ceiling near mid-90s without a better multi-objective preference recipe.

### 6.5 Residency and serve throughput (measured)

#### Single-stream (denser-C L3T2, n=256 greedy, reference Ryzen 2×32 MiB L3)

| Metric | Soft `--l3-resident` | Hard `--residency-hard` + FIFO |
|--------|---------------------:|--------------------------------:|
| C runtime WS | 24.32 MiB | 24.32 MiB |
| L3T2 weights payload | 20.27 MiB | 20.27 MiB |
| On-disk `.ll3m` | 20.32 MiB | 20.32 MiB |
| Fits 32 MiB CCD | **YES** | **YES** |
| mincore mlock (model+state) | **100%** | **100%** (mlockall OK) |
| Decode throughput | ~**260–270** tok/s (CCD multi-core soft) | ~**280–282** tok/s (CPU pin + `SCHED_FIFO`) |
| Early unisolated logs (pre-FIFO fix) | 293.6 tok/s · DRAM 23.9% | 268.6 tok/s · DRAM 26.6% |
| L2 fills from local L3 (best hard) | — | up to ~**97%** |
| L2 fills from DRAM (best hard) | often ~30–40% soft | **GOOD ≤5%** on long windows with FIFO; else WEAK |
| isolcpus | no | no (boot still optional) |
| SCHED_FIFO | no | **OK** with `cap_sys_nice` |

**Takeaway:** size-fit and mlock are demonstrated. On this host, **hard+FIFO is the preferred single-stream path** for both throughput and PMU honesty once capabilities are set; soft multi-core on the CCD does not raise tok/s. Absolute zero-DRAM still benefits from boot `isolcpus` / IRQ steering (`docs/L3_RESIDENCY.md`). Design estimator (30.22 MiB) remains the freeze gate; C runtime is leaner (~24.3 MiB).

#### Multi-session serve under the same CCD gate

Shared weights + independent states: `WS = weights + N × state(kv_cap)`. Tool: `c_infer/multi_bench`.

| Config | WS | Per-stream | **Aggregate** |
|--------|---:|-----------:|--------------:|
| N=1, kv=256 (full T) | 24.3 MiB | ~280 | ~280 |
| **N=2, kv=256** (max full context) | 28.4 MiB | ~274 | **~550** |
| N=3, kv=256 | 32.4 MiB | — | **refused** (over gate) |
| **N=8, kv=128** | 29.6 MiB | ~270 | **~2100** |
| **N=8, kv=64** (best balanced multi) | 25.2 MiB | ~277 | **~2190** |
| N=16, kv=32–48 (peak aggregate) | ~26–28 MiB | ~146 | **~2300–2320** |

Max **N** under gate: **2** at full T=256; **8** at kv=128 (one physical core each on this CCD); up to **24** only with very short KV. N≤8 scales near-linearly (~0.97–0.99); N=16 peaks aggregate but halves per-stream (SMT). Artifacts: `checkpoints/compare/l3_isolation_bench/N_SESSION_MAX.md`.

---

## 7. Discussion

### 7.1 Is 41% “good”?

Relative to **random** and to a prior 40.1% tool path, denser continue is a **real but modest** closed-book gain. Relative to **Qwen 0.8B**, a ~13 pp macro gap is expected: Qwen has ~10× parameters, a larger vocabulary, vastly more pretraining compute, and no L3 residency constraint. For an **L3-resident agent**, the relevant question is whether language + tool policy + host facts deliver reliable behavior in a tiny pack. Product-gated scores (**101/101**) support that framing more than pure gate-off (**90/101**), with the caveat that the suite is closed and small.

### 7.2 Where capacity is spent

ARC-Easy remains near chance: school science is not free under 79M ternary weights. SciQ is stronger when dense educational signal is preserved (C/C+/MIX). C+ shows capacity can be **reallocated** (SciQ/ARC up, BoolQ down) without macro progress. OpenBookQA is weak for both systems but worse for LL3M (and below chance). Investing in **teacher-generated atomic curricula** and **KD from 0.8–1.7B teachers** into the same shape is more promising than packing Wikipedia into ternary weights. Pure no-call after tool SFT likely needs **preference/RL on false_call**, not more short CE thrash.

### 7.3 Knowledge density hierarchy

Information per “product byte” (descending):  
**host KV / triples** ≫ **atomic train cards** ≫ **short edu prose** ≫ **raw web** ≫ **ternary weight bits for rare facts**.

System capability is \(f(\text{weights},\;\text{KB},\;\text{tools},\;\text{gate})\), not SciQ alone.

### 7.4 Systems claim discipline

We separate (i) **train** scale (GPU/DRAM free), (ii) **serve pack** size (~20.3 MiB file), (iii) **infer WS** under L3 (design 30.2 / runtime 24.3 MiB single-stream), (iv) **PMU DRAM share** (GOOD achievable with FIFO; WEAK without), (v) **agent policy flags** (assist/gate), and (vi) **multi-session density** (N concurrent states under the same CCD gate). Marketing “small model,” “L3-resident,” “100% agent,” or “tok/s” without stating which of these applies is misleading; we state them explicitly.

### 7.5 Multi-session as a first-class CCD product mode

The gate is not only a single-stream size limit: it is a **session-packing** limit. Full product context admits **N=2**; shortening KV (still natural for tool turns) admits **N=8** near-linear multi-serve at ~2k aggregate tok/s without leaving the CCD. That is a systems consequence of MQA + ternary weights + host tools, not a separate product line.

### 7.6 What would strengthen the claim further

Boot `isolcpus`-grade DRAM% toward ~0; same-shape fp32 MC; bitnet.cpp / llama.cpp latency peers; KB on/off ablation; BFCL-style held-out tools; multi-seed MC; GRPO-lite on false_call under QAT; production scheduler over `multi_bench`.

---

## 8. Limitations

1. **Closed-book MC** is not the product objective; agent and residency matter more.  
2. Context **256** and vocab **2048** limit long documents and English token efficiency; LL ranking vs Qwen is not iso-tokenizer.  
3. Teacher KD and multi-trillion-token overtraining were **not** used in the reported ladder.  
4. Qwen baseline is weight-only int4 Base, not an identical train recipe; comparison is **capability-under-size**, not iso-data. NF4 and public ~100M SLMs not yet in the table.  
5. Sample-level fluency remains imperfect; MC does not guarantee good free-form prose.  
6. Hard residency with boot `isolcpus` is environment-dependent; with `cap_sys_nice` FIFO we can reach **GOOD** DRAM% on long windows, but without isolation PMU remains **WEAK**.  
7. **No single pure-policy pack** jointly maximizes tools, SciQ, and pure no-call under short QAT; dual ship / product gate is intentional.  
8. Agent suite is **closed (101 cases)**, not BFCL; product scores include host gate.  
9. Preference optimization against false_call and large-teacher tool-timing KD remain future work.  
10. Full wall-clock GPU-hours / energy for every stage are not yet published; train hosts differ from the serve reference CPU.  
11. Single seed (42) for MC; no confidence intervals.  
12. Design WS estimator and C runtime WS differ (~30.2 vs ~24.3 MiB); both fit the gate, but accounting must stay explicit.  
13. Multi-session max N depends on `kv_cap`; full T=256 admits only N=2 under the 30.4 MiB gate. Aggregate peak uses short KV and SMT.

---

## 9. Ethics and dual-use

A ~20 MiB pack can run offline on commodity CPUs, which is useful for privacy and edge deployment but also lowers the bar for unsupervised local generation. The closed tool registry and optional host gates reduce open-ended tool abuse relative to unrestricted browser agents. We do not release unsafe tool backends. Data licenses for FineWeb-Edu / synthetic cards and a model release statement will accompany any public weight drop.

---

## 10. Conclusion

LL3M shows that a **~79M ternary** model can be co-designed with a **single L3 CCD** inference gate, a **portable C** pack, continuous **QAT**, and **host-side knowledge**. On standard MC benchmarks denser continue trails Qwen3.5-0.8B weight-only int4 by about **13 percentage points** (**23.6%** relative) at macro while using far fewer parameters and an L3-class footprint. Further dense continue can raise SciQ without raising macro; short tool QAT exposes a **policy–knowledge Pareto**. Tool-only SFT without CE replay harms closed-book science and pure agent scores; multi-mix restores SciQ but not pure no_call. Product agents should report **gated** scores; pure gate-off is a stress test. On the tool ship pack, product is **101/101** and pure is **90/101**. Size-fit and mlock are measured; hard+FIFO single-stream holds ~**280–282 tok/s** with optional **GOOD** DRAM%; multi-session packing under the same gate reaches ~**2k tok/s** aggregate at N=8 (short/medium KV) or ~**550** at full T=256 N=2. The path forward is **host KB growth, honest multi-flag agent metrics, multi-mix or preference learning, multi-session serving, isolcpus-grade residency proof, and optional KD**, not encyclopedia-in-weights or inflated pure-policy claims.

---

## Acknowledgments

Sole author: **Marcel Wysocki**. Thanks to the open ecosystems this work builds on (FineWeb-Edu / Hugging Face datasets, Qwen baselines, BitNet-style ternary training literature, and related open tooling).

---

## References

See [`ll3m_l3_resident.bib`](ll3m_l3_resident.bib) (BitNet / b1.58 / bitnet.cpp, Engram, Phi/SmolLM/FineWeb, Qwen3, When2Call/BFCL, MobileLLM, llama.cpp/MLC/PowerInfer, MQA/GQA, QAT). Selected anchors:

1. Wang et al., BitNet, arXiv:2310.11453; Ma et al., b1.58, arXiv:2402.17764; bitnet.cpp, arXiv:2502.11880.  
2. DeepSeek, Engram, arXiv:2601.07372.  
3. Gunasekar et al., Textbooks Are All You Need; Allal et al., SmolLM2; Penedo et al., FineWeb.  
4. Qwen team, Qwen3 technical report arXiv:2505.09388; Qwen3.5-0.8B-Base model card.  
5. When2Call arXiv:2504.18851; Gorilla/BFCL; SLM tool-calling evaluations.  
6. MobileLLM arXiv:2402.14905; llama.cpp; MLC-LLM; PowerInfer.  
7. Shazeer MQA; Ainslie GQA; PaLM z-loss; DeepSeek MTP.

---

## Appendix A: Hyperparameters (dense continue C)

| Knob | Value |
|------|-------|
| Resume | `v76_base_en` step 60000, QAT on |
| Steps | 90000 absolute (+30000) |
| Optimizer | Lion |
| LR / warmup | 1e-4 / 500 |
| Microbatch × accum | 8 × 8 (16384 tok/step) |
| Dropout | 0.1 |
| Device | CUDA (train host `spark1`) |
| Serve quant | ternary + int8 embed QAT |

## Appendix B: Multi-mix SFT

| Knob | Value |
|------|-------|
| Base | `v76_dense_c` |
| Mix | 50% dense / 20% fact / 30% tool (char budget) |
| Steps | +5000, lr 3e-5 |
| QAT | ternary + int8 |

## Appendix C: Reproducibility artifacts (repository)

| Artifact | Path (project) |
|----------|----------------|
| Freeze | `checkpoints/v75_freeze.json` |
| Dense ladder doc | `docs/DENSE_KNOWLEDGE_LADDER.md` |
| Residency protocol | `docs/L3_RESIDENCY.md` |
| C MC / D / MIX | `checkpoints/v76_dense_c/std_benchmarks_expC_limit400.json`, `checkpoints/v76_dense_tool/std_benchmarks_expD_limit400.json`, `checkpoints/v76_dense_mix/std_benchmarks_expMix_limit400.json` |
| C+ full MC | `checkpoints/v76_dense_c_p1/std_benchmarks_limit400.json` |
| Aggregate vs Qwen int4 (incl. C+) | `checkpoints/compare/ll3m_vs_qwen35_family_int4.json` |
| Qwen int4 family | `checkpoints/compare/Qwen__Qwen3.5-*_int4_limit400.json` |
| Residency logs | `checkpoints/compare/residency_{soft,hard}_v76_dense_c.txt` |
| Multi-session sweep | `checkpoints/compare/l3_isolation_bench/N_SESSION_MAX.md`, `n_session_sweep_v2_clean.csv` |
| Multi-stream binary | `c_infer/multi_bench` (`--streams`, `--kv-cap`) |
| C decode micro-opts log | `autoresearch.md` / `autoresearch.jsonl` |
| QAT vs post-hoc val CE | `checkpoints/v75_ternary/quality_compare.json` |
| Agent product/pure ship | `checkpoints/v76_dense_c_pure_v2/battery_product_gate_on.json` (**101/101**), `…/battery_no_assist_gate_on.json` (**99/101** prior), `…/battery_true_pure_greedy.json` (**90/101**); pins in `checkpoints/paper_freeze_v072/` |
| Paper freeze (v0.7.2) | `checkpoints/paper_freeze_v072/MANIFEST.json` (KB 1022/3057 + agent pins) |
| Agent denser-C pure | `checkpoints/v76_dense_c/battery_true_pure_greedy.json` (42/101; basename collides with pure_v2) |
| Agent D / MIX pure | `checkpoints/v76_dense_tool/battery_D_no_assist.json`, `checkpoints/v76_dense_mix/battery_MIX_no_assist.json` |
| Dense mix manifest (161M tok) | `data/base_en_dense/manifest.json` (bins may be train-host only; see note) |
| Dense C progress | `checkpoints/v76_dense_c/progress.json` (reconstructed from pipeline log if rsync incomplete) |
| Dense pipeline log | `checkpoints/dense_pipeline/pipeline.log` |
| Golden train–serve | `checkpoints/v76_dense_c_pure_v2/golden.log` |
| Quarantined bad battery | `checkpoints/v76_dense_c/_quarantine/` |
| C runtime | `c_infer/` |
| Pack accounting | design WS ~30.22 / C payload 24.32 (weights payload 20.27) / on-disk `.ll3m` 20.32 MiB |

## Appendix D: Suggested venue framing

| Venue style | Pitch |
|-------------|--------|
| **Systems / Edge ML** | L3-resident WS + C serve + residency protocol (size-fit now; DRAM% roadmap) |
| **Efficient ML / Quant** | Ternary+int8 native QAT lock to C pack |
| **Agents / SLM tools** | Engram-lite host KB + When2Call-style SFT + multi-mix finding |
| **Workshop (Tiny / on-device)** | Full product narrative under 32 MiB L3 |

Primary claim for review: **co-design of model, quant, cache gate, and host knowledge**, not SOTA MMLU.

## Appendix E: Author checklist before arXiv

- [x] Author: Marcel Wysocki (sole)  
- [ ] Affiliation / email (optional)  
- [x] Expand BibTeX with primary arXiv IDs  
- [x] Fix MD/TeX Qwen per-task sync (portable int4)  
- [x] Report D/MIX pure agent batteries  
- [x] On-host residency measurement (size PASS, DRAM% WEAK)  
- [ ] isolcpus-grade DRAM% target run  
- [ ] Wall-clock train hours / energy  
- [ ] Optional: human eval of short English samples  
- [ ] License and model release statement  
- [x] LaTeX draft synced (v0.6)  
- [x] Ethics statement present  
- [x] Product test freeze = L23×d576 (`tests/test_product_dense.py`)  
- [x] Pack payload vs on-disk file wording  
- [x] Full agent battery paths (no bare basename collisions)  
- [x] Family aggregate includes C+  


---

*End of draft v0.7.9*
