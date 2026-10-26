# Semantic compressor (research)

**Date:** 2026-08-10 (theory §1b added same day)  
**Status:** P0/P1 scaffold live (`ll3m/semcomp/`); research design for full ladder  
**Goal:** a **tool** that compresses *training corpora* so the model sees fewer tokens while preserving **all task-relevant meaning** for learning.  
**Related:** [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md) · [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) · [BASE_ENGLISH.md](BASE_ENGLISH.md) · [L3_CAPACITY_SEARCH.md](L3_CAPACITY_SEARCH.md) · [QUANTIZATION.md](QUANTIZATION.md)

---

## 0. One-line thesis

**Compress the *surface form* of training data (redundancy, style, near-duplicates, filler) while leaving the *semantic payload* (facts, relations, procedures, tool traces, discourse structure the model must learn) recoverable under an explicit fidelity contract.**

“100% of meaning” is a **product claim over a defined meaning model**, not a free lunch against Shannon or Kolmogorov. The tool must **state what meaning is**, **measure it**, and **refuse to drop** anything outside the residual error budget.

**Capability thesis (LL3M):** under a fixed L3 param pack and fixed train steps, semantic compression is one of the main ways to **raise unique knowledge signal per token**, i.e. to *cram more of a curriculum into the same model and wall time*, without growing the serve working set. It does **not** expand how many independent bits the weights can store.

---

## 1. Why this exists (LL3M pressure)

| Constraint | Effect on data |
|------------|----------------|
| Serve WS ≤ one L3 CCD (~30.4 MiB) | Params capped (~79M ternary, ~195M binary research); **data quality density** is the main free variable |
| Long trains are GPU-time scarce | Fewer tokens that still teach → higher capability per step |
| Stories / web are bloated | Same fact restated in many surface forms burns capacity |
| Dense ladder already half-does this | Atomic cards, MinHash, short FineWeb, Engram facts: **manual / pipeline** semantic density |

Semantic compression is the **generalized, automatic** version of Exp B (atomic mix + near-dedup): turn a messy corpus into a **minimal equivalent curriculum**.

Mental model:

```
raw corpus (high tokens / low unique claims)
        │  semantic compressor
        ▼
dense corpus (low tokens / same claim set + structure)
        │  BPE / train
        ▼
better CE / SciQ / agent per wall-hour under same L3 model
```

This is **not** weight compression (ternary/binary). It is **data-plane** compression *before* train.

---

## 1b. Can we cram more knowledge into the model?

**Short answer: yes for curriculum density and sample efficiency; no for unbounded parametric encyclopedias.**

### 1b.1 Two planes (do not confuse them)

| Plane | What it is | LL3M lever | Bounds “knowledge” how? |
|-------|------------|------------|-------------------------|
| **Weight plane** | Bits in serve pack (ternary/binary Linears + int8 embeds + KV) | Shape search, quant alphabet, C residency | Hard cap ≈ **CCD** (~30.4 MiB product gate). Order **tens of MiB**, not GiB. |
| **Data plane** | Tokens the optimizer sees over a train run | Mix quality, dedup, **semcomp**, curriculum stages | Cap is **steps × tokens/step × signal density**. Soft; improved by denser data. |
| **Host plane** | Engram / `lookup` / wiki / tools | KB v2, tool SFT, agent loop | Long-tail facts **outside** weights; system capability without growing L3 pack. |

Phi / SmolLM / Qwen-small “know a lot” largely because they sit on **orders of magnitude more weight bits** plus trillion-token curated data, not because surface compression alone invents capacity. See [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md).

```text
┌─────────────────────────────────────────────────────────┐
│  System knowledge (user-visible)                        │
│                                                         │
│   parametric  ── weights under L3 (language + core      │
│                  school/tool policy + some facts)       │
│                                                         │
│   host        ── Engram / tools (long-tail, live data)  │
│                                                         │
│   train data  ── what gets *into* parametric memory     │
│                  (semcomp densifies this path)          │
└─────────────────────────────────────────────────────────┘
```

Semantic compression acts on the **train data → parametric** arrow. It does not enlarge the weight box and does not replace the host box.

### 1b.2 What “more knowledge” can mean (define before measuring)

| Sense | Definition | Does semcomp help? |
|-------|------------|--------------------|
| **K1 Curriculum coverage** | Fraction of a fixed claim set $M^\star$ the train run *presents* | **Yes:** more unique claims per token / per hour |
| **K2 Parametric recall** | Closed-book MC / freeform on claims that fit in capacity | **Yes if** claims were bottlenecked by data density and model has spare capacity |
| **K3 World knowledge** | Open-ended encyclopedia / long-tail | **No alone:** weight bits too few; use host plane |
| **K4 System capability** | Agent + tools + language under L3 | **Indirectly:** denser tool episodes + fact cards + language replay |

Product-relevant senses for LL3M: **K1 + K2 + K4**. Not K3 as a weight-only goal.

### 1b.3 Why theory says denser data helps under a param wall

Fix architecture, quant, steps $S$, tokens/step $T_{\text{step}}$. Rough teaching budget:

\[
B_{\text{teach}} \approx S \cdot T_{\text{step}} \cdot \rho
\]

where $\rho$ is **unique useful claim density** (claims / token, or nats of task-relevant information / token).

| Intervention | Effect on $B_{\text{teach}}$ | Effect on serve WS |
|--------------|------------------------------|--------------------|
| Longer train | ↑ $S$ (cost GPU time) | none |
| Bigger batch | ↑ $T_{\text{step}}$ (VRAM / wall) | none |
| **semcomp** | ↑ $\rho$ (or same claims at lower $S$) | **none** |
| Binary reinvest (~195M) | more capacity to *store* densified claims | stays under C-calibrated gate |
| Ternary shape thrash | small param Δ at T=256 | already near est ceiling |

So under L3, when you cannot freely grow params, **raising $\rho$ is the free variable** that still moves knowledge *into* the weights. That is the theoretical case for semantic compression as a capability tool, not only a disk saver.

**Capacity caveat:** if the model is already saturated on a claim class (memorization ceiling for that bit budget), denser data will not raise K2 further. Then you need more params (binary reinvest), better quant alphabet, or offload (host). Semcomp still helps K1 presentation and multi-task mixes (facts + tools + English) by reducing waste.

### 1b.4 What semcomp cannot do (honest bounds)

| Claim | Reality |
|-------|---------|
| “Lossless meaning for all human readings” | Only lossless w.r.t. versioned $M^\star$ |
| “Unlimited facts in 30 MiB ternary” | Violates weight-plane budget; Engram for long-tail |
| “Gzip the corpus → smarter model” | Bit-lossless text still costs full tokens after decode; no ranking of *teaching* value |
| “Higher $\rho$ always raises SciQ” | Only if SciQ claims are in $M^\star$, optimized, and not capacity-saturated |
| “Replaces multi-mix / tool SFT” | Different skills; compress episodes, do not delete tool channel |

### 1b.5 Complementarity table (weights × data × host)

| Goal | Weights (quant/shape) | Data (semcomp / mix) | Host (Engram/tools) |
|------|----------------------|----------------------|---------------------|
| English fluency | capacity for LM head / layers | mixed render, not pure atomic | low |
| School / SciQ-style facts | capacity + QAT | **atomic / QA densify** | optional cards |
| Tool policy | residual + SFT | densify episodes, keep format | runtime tools |
| Long-tail names / live data | **avoid packing** | optional short cards | **primary** |
| Agent battery | policy in weights | multi-mix | gate + tools |

**Principle (repeat):**  
semcomp ↑ **teaching bits per token**; quant ↑ **params per MiB**; Engram ↑ **facts per host byte**. All three raise *system* capability under L3. None alone is “pack Wikipedia into CCD.”

### 1b.6 Worked picture (fixed 50k-step run)

Suppose a messy mix has $\rho_{\text{raw}} = 0.02$ unique hard claims per token, and semcomp yields $\rho_{\text{sc}} = 0.05$ at the same claim set (or larger claim set at same token budget).

| | Raw | Semcomp |
|--|-----|---------|
| Tokens seen | $S \cdot T_{\text{step}}$ | same or fewer |
| Unique claims presented | $\propto \rho_{\text{raw}}$ | $\propto \rho_{\text{sc}}$ (~2.5× if ratio holds) |
| Serve WS | fixed | fixed |
| GPU hours | baseline | baseline or **lower** for same claim coverage |

Tiny smoke (see §15): 40 high-value blocks from cap_mix → BPE ratio **~0.44** mix/slice with claim-list gate ok. That is a lower bound illustration on already-filtered QA/tool text; story-heavy web should compress harder on $\rho$.

### 1b.7 How to prove it on LL3M (experiment contract)

Hold **shape + quant + steps + tokenizer** fixed (e.g. freeze ternary or v78 binary one-at-a-time).

| Arm | Data | Expect |
|-----|------|--------|
| A0 | raw / story-heavy | worst claims/hour |
| A1 | exact + MinHash only (today dense ladder) | baseline |
| A2 | **semcomp atomic** on same sources | ↑ SciQ or ↑ claims@fixed steps; val CE on atomic may look different |
| A3 | semcomp mixed (atomic + compact_prose + tool episodes) | better English samples than A2; hold agent |
| A4 | A3 + Engram host unchanged | system K4; do not attribute host hits to weights |

**Primary metrics:** SciQ@400 (or claim probe set derived from $M^\star$), agent battery, unique claims presented, wall hours.  
**Secondary:** val CE on *original* surface (may worsen while probe improves: distribution shift, not always a bug).  
**Gate:** $M^\star$ coverage = 100%, invention = 0 on the compressed train set.

**Success for “more knowledge in the model” (K2):** probe accuracy ↑ at fixed steps without battery regression, with host tools **off** for the probe.  
**Success for sample efficiency (K1):** same probe accuracy at **fewer** steps or fewer tokens.

### 1b.8 Interaction with current trains (v78 binary, dual ship)

| Track | Role of semcomp |
|-------|-----------------|
| **v78 binary ~195M** | Extra weight capacity; densified data may fill capacity that stories leave empty. Fair SciQ twin should use densified SciQ/fact mix, not base-only. |
| **Ternary ship ~79M** | Less parametric room; semcomp + Engram matter *more* for system knowledge. |
| **SciQ denser stages** (`cap_mix_v16_*`) | Hand-rolled weak semcomp (atomic-ish cards). Automating this is the productization path. |
| **Agent multi-mix** | Compress tool episodes (keep `<call>` structure); drop system-prompt spam and near-dup tool traces. |

### 1b.9 Decision summary (knowledge)

| Question | Answer |
|----------|--------|
| Can semcomp cram more knowledge into the *training run*? | **Yes** (higher $\rho$, K1) |
| Can it raise closed-book recall for claims in capacity? | **Often yes** (K2), measure with probes |
| Can it turn L3 pack into a world encyclopedia? | **No** (weight plane + Engram) |
| Does it grow serve WS? | **No** |
| Best paired with? | Binary/ternary pack as-is, multi-mix stages, host KB for long-tail |
| Ship bar for the claim “more knowledge”? | Fixed-step probe ↑, $M^\star$ coverage 100%, battery non-regress |

---

## 2. What “100% of meaning” can mean (be precise)

Meaning is not one thing. Pick a **meaning model** $M$ and require:

\[
M(\text{compress}(D)) \;\supseteq\; M(D)
\quad\text{or}\quad
d_M\bigl(M(D),\, M(\text{compress}(D))\bigr) = 0
\]

within a declared measure $d_M$.

### 2.1 Candidate meaning models (choose per corpus)

| Layer | What is preserved | Typical extractors | Failure if dropped |
|-------|-------------------|--------------------|--------------------|
| **M1 Lexical** | Token multiset / rare terms | set of content words | Named entities, units |
| **M2 Propositional** | Atomic facts / triples | open IE, LLM fact extract, KB cards | SciQ / closed-book |
| **M3 Relational** | Graph of entities + edges | coref + relation extract | Multi-hop |
| **M4 Procedural** | Steps, tool calls, pre/post conditions | episode parse (`<call>…</call>`) | Agent battery |
| **M5 Discourse** | Order constraints (cause, contrast) | RST / light discourse tags | Reasoning traces |
| **M6 Style / voice** | Register, dialect, story cadence | optional; **usually droppable** for knowledge train | Creative writing only |

**LL3M default product meaning** (recommended):

- **Must keep (hard):** M2 + M4 + named entities / numbers / units from M1  
- **Soft keep:** M3, M5 when present and cheap to verify  
- **May drop:** M6, boilerplate, ads, near-paraphrase duplicates, pure narrative color that carries no new proposition

So “100% meaning” for us means:

> **100% of the hard meaning model $M^\star$ (propositions + tool episodes + critical literals), zero silent loss; soft layers only with explicit flags.**

Anything else is marketing.

### 2.2 What cannot be guaranteed

| Claim | Reality |
|-------|---------|
| Lossless for *all* human interpretations | Impossible; interpretation is open-ended |
| Bit-identical decode of original text | That is gzip, not semantic compression |
| Same next-token CE on original surface | Compressed text is a *different* distribution by design |
| Teacher LLM extractors never err | Extractors **define** $M$; error in extract = error in “meaning” |

The honest contract: **lossless w.r.t. $M^\star$ under a fixed extractor version**, with adversarial tests and a kill switch when residual risk is nonzero.

---

## 3. Related work (is there research on this?)

**Yes.** Nobody ships a product named “semantic compressor with 100% meaning,” but several mature lines of research attack the same goal from different angles: **remove redundant or low-value surface form**, **keep high-value teaching signal**, sometimes **rewrite into denser textbooks**. Our design sits at the intersection of **semantic dedup**, **quality filtering**, **atomic claim extraction**, and **selective training**.

### 3.1 Map: families → steal / skip for LL3M

| Family | Idea | Steal for `semcomp`? | Canonical refs (§14) |
|--------|------|----------------------|----------------------|
| **Semantic / embedding dedup** | Drop near-paraphrase docs via embedding similarity | **Yes** (doc-level stage; claim-level is stricter) | SemDeDup [1], D4 [2], NeMo Curator SemDedup [3] |
| **Exact / MinHash near-dedup** | Hash or Jaccard on n-grams | **Yes** (already in dense ladder) | FineWeb pipeline [8]; our MinHash |
| **Quality / edu filters** | Score docs, keep educational subset | **Yes** stage-0 gate | FineWeb-Edu [8], DataComp [9], DCLM [10] |
| **Textbook / synthetic densify** | Generate dense educational surface from web or pure synth | **Yes** as render *target* style; invent risk | Phi textbooks [11][12], Cosmopedia [13] |
| **Token-level selection** | Train loss only on “useful” tokens | **Related** (orthogonal: train-time, not corpus rewrite) | Rho-1 SLM [14] |
| **Instruction data selection** | Keep 5% of SFT that matches a skill | **Yes** for tool/agent mixes | LESS [15] |
| **Atomic fact units** | Decompose text into checkable propositions | **Yes** core of $M^\star$ and verify | FActScore [16], Claimify [17], AFEV [18] |
| **Structured IE from text** | LLM → JSON / triples from papers | **Yes** for extract stage | LLM-NERRE [19] |
| **Host / conditional memory** | Facts outside residual stream | **Complement** not replace | Engram [20] |
| **Classic abstractive summary** | ROUGE / shorten for humans | **No** as sole gate (lossy by design) | — |
| **Weight compression** | Prune / quant model, not data | **Orthogonal** (our pack path) | — |

### 3.2 What the literature already shows (takeaways)

1. **Semantic near-dup is real and large at web scale.** SemDeDup [1] removes semantically similar but non-identical samples using pretrained embeddings; models train faster and often match or beat full-data quality. D4 [2] combines de-duplication with diversification in embedding space for LLM pretrain efficiency. Production stacks (NVIDIA NeMo Curator [3]) ship this as a standard curation stage.

2. **Filtering beats “more raw tokens” for knowledge/reasoning.** FineWeb-Edu [8] and DCLM [10] show educational / curated subsets outperform larger dirty crawls on MMLU-class and multi-task suites at fixed train compute. That is the same $\rho$ argument as §1b, at document granularity.

3. **Dense synthetic surface (textbooks) works for small models.** Phi [11][12] and Cosmopedia [13] deliberately replace messy web with short, high-density educational text. Closest published cousin of our **render** stage, but usually *generative* (risk of invention) rather than *extract-then-verify*.

4. **Not all tokens / examples are equal.** Rho-1 [14] drops gradient on low-value tokens *during* training; LESS [15] selects a tiny SFT subset that beats full data for a target skill. Complementary to rewriting the corpus offline.

5. **Atomic facts are a standard meaning unit.** FActScore [16] evaluates LMs by atomic fact precision; Claimify [17] and AFEV [18] extract claims for verification. We reuse that unit as $M^\star$ for *training corpus* contracts, not only eval.

6. **Gap our design fills.** Prior work mostly: (a) **drop** documents, (b) **filter** by score, (c) **synth** new text, or (d) **mask** tokens at train time. Few systems publish a full loop of **extract claims → claim-level dedup → deterministic re-render → coverage/invention gate** aimed at small L3-resident models. That loop is LL3M-specific productization of the above lines.

### 3.3 Anti-patterns (literature-aligned)

- Pure abstractive summary without claim coverage check (classic summarization metrics ≠ $M^\star$)  
- Aggressive LLM rewrite without span grounding (Phi-style synth can invent; we require source link or refuse)  
- Treating embedding cosine / Jaccard alone as “same meaning” (SemDeDup clusters can still merge distinct propositions; use claim keys for hard facts)  
- Confusing **data** compression with **weight** compression (pruning/quant papers)  

### 3.4 Positioning one-liner

> **SemDeDup/D4/FineWeb-Edu densify by selection; Phi densifies by synthetic rewrite; Rho-1 densifies by selective loss; we densify by claim-preserving rewrite with a verify gate**, under an L3 param wall where $\rho$ is scarce.

---

## 4. Formal problem

### 4.1 Inputs / outputs

```
Input:  corpus C = {d_i}  (docs / stories / SFT episodes)
        meaning model M*  (versioned extractor + schema)
        budget B          (max tokens, or max bytes, or max claims)

Output: corpus C' = {d'_j}
        map file      (claim_id → source spans)
        report        (coverage, residual risk, compression ratio)
```

### 4.2 Objectives

1. **Coverage:** every hard claim in $M^\star(C)$ appears in $M^\star(C')$ (or linked host KB).  
2. **No invention:** no hard claim in $M^\star(C')$ that is unsupported by $C$ (or by approved external KB with citation).  
3. **Token efficiency:** minimize $\sum |d'_j|$ (BPE tokens under product tokenizer).  
4. **Train utility (soft):** improve or hold val CE / SciQ / agent under fixed step budget vs raw $C$.

Primary hard constraints are (1)(2). (3)(4) are optimizers.

### 4.3 Compression ratio

\[
R_{\text{tok}} = \frac{\text{tokens}(C')}{\text{tokens}(C)}, \qquad
R_{\text{claim}} = \frac{|M^\star(C')|}{|M^\star(C)|}
\]

Success target for research:

- $R_{\text{claim}} = 1.0$ (hard claims)  
- $R_{\text{tok}} \ll 1$ (e.g. 0.2–0.5 on web/story mixes; lower on already-atomic cards)  
- train metric non-regression under matched token budget

---

## 5. Architecture (proposed tool)

Name (working): **`semcomp`** (CLI + library under `scripts/` / `ll3m/` later).

```
┌─────────────────────────────────────────────────────────────┐
│                     semcomp pipeline                          │
│                                                               │
│  C ──► [0 normalize] ──► [1 segment] ──► [2 extract M*]       │
│              │                              │                 │
│              │                              ▼                 │
│              │                     claim store (canonical)    │
│              │                              │                 │
│              ▼                              ▼                 │
│         [3 dedup claims] ◄──────── [link / merge / conflict]  │
│              │                                                │
│              ▼                                                │
│         [4 render]  atomic cards / compact prose / episodes   │
│              │                                                │
│              ▼                                                │
│         [5 verify]  coverage + no-invention + lit checks      │
│              │         fail → refuse drop / re-render         │
│              ▼                                                │
│         C' + manifest.json + coverage report                  │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 Stage 0: Normalize

- Encoding, whitespace, boilerplate strip (nav, cookie banners)  
- Language filter (product: English)  
- Optional PII scrub (policy)  
- **Do not** rewrite content here

### 5.2 Stage 1: Segment

Units of work:

| Unit | Use |
|------|-----|
| Document | web / FineWeb |
| Story / paragraph | TinyStories-style |
| Episode (user/assistant/tool) | tool SFT |
| Sentence / clause | atomic extraction grain |

Preserve **source IDs** for audit.

### 5.3 Stage 2: Extract $M^\star$ (heart of “meaning”)

**Hard schema (v1):**

```json
{
  "claim_id": "sha1...",
  "type": "fact | procedure | tool_trace | definition | constraint",
  "subject": "...",
  "predicate": "...",
  "object": "...",
  "literals": {"numbers": [], "units": [], "entities": [], "negation": false},
  "source_spans": [{"doc": "...", "start": 0, "end": 0}],
  "confidence": 0.0,
  "extractor": "rule|llm|hybrid@version"
}
```

**Extractor ladder (prefer cheap → expensive):**

1. **Rules / regex:** dates, units, `A is B`, tool call XML  
2. **Template cards** from host Engram / SciQ / KB v2  
3. **Local SLM / teacher** only with *grounded* mode: extract, then **span-link** back to source text  
4. Never free-form “summarize the page”

**Tool episodes (M4):** parse `<call>…</call>` and results into structured traces; compress result *payloads* only if a typed reducer exists (e.g. wiki → first definition sentence + URL), not free summary.

### 5.4 Stage 3: Claim-level dedup and merge

Operate on **claims**, not only documents:

| Op | Rule |
|----|------|
| Exact claim hash | drop duplicates |
| Soft match (embedding / MinHash on claim text) | merge if same (s,p,o) + literals |
| Conflict | keep both with `conflict=true` or prefer higher-confidence + source count; **never silent drop** |
| Paraphrase clusters | one canonical surface; retain alias list for train spice if desired |

This is stricter than doc-level MinHash (Jaccard 0.85 can kill a unique fact embedded in similar prose).

### 5.5 Stage 4: Render (surface form of compressed corpus)

Several **render profiles** (same claim set, different surfaces):

| Profile | Form | Tokens | When |
|---------|------|--------|------|
| **`atomic`** | One claim per line / short card | lowest | knowledge pretrain, SciQ twin |
| **`compact_prose`** | Dense paragraphs packing N claims | medium | base English fluency |
| **`qa`** | Q→A from claims | medium | closed-book SFT |
| **`episode`** | Minimal tool dialogues | low–med | agent |
| **`mixed`** | Curriculum mix of above | — | product ladder |

Rendering is **deterministic given claim set + seed** so runs are reproducible.

Example (atomic):

```text
# claim:c9f3… src:doc42
The capital of France is Paris.
# claim:a12b… src:doc42
Water freezes at 0 degrees Celsius at standard pressure.
```

### 5.6 Stage 5: Verify (the “100%” gate)

Before writing $C'$:

1. **Coverage:** $|M^\star(C) \setminus M^\star(C')| = 0$ for hard claims (match by canonical key).  
2. **No invention:** every claim in $C'$ span-links to $C$ or approved KB.  
3. **Literal audit:** all numbers/units/entities from source claims appear in render (string or normalized form).  
4. **Negation / quantifier check:** “not”, “never”, “all”, “some” preserved when in claim.  
5. **Round-trip sample:** human or second-model judge on stratified sample; fail batch if error rate > ε.

If verify fails: **do not emit lossy $C'$**. Emit residual report and keep original spans for failed items (partial compress).

---

## 6. Metrics and experiments

### 6.1 Offline (no train)

| Metric | Pass |
|--------|------|
| Claim coverage | = 100% hard |
| Invention rate | = 0 |
| Token ratio $R_{\text{tok}}$ | report; lower better |
| Unique claim density | claims / 1k tokens ↑ |
| Extractor agreement (two extractors) | high on sample |
| Human audit (n≥100 claims) | 0 critical misses |

### 6.2 Online (train ablations)

Fix **architecture + steps + tokenizer**. Vary only data:

| Arm | Data |
|-----|------|
| A0 | raw mix |
| A1 | exact + MinHash only (today) |
| A2 | **semcomp atomic** |
| A3 | semcomp mixed render |
| A4 | A2 + host Engram unchanged |

Measure: val CE, SciQ@400, agent battery, tok/s wall, tokens seen.

**Hypothesis:** A2 ≥ A1 on SciQ/claims per hour; A3 better language samples than pure atomic; A0 worst density.

### 6.3 Adversarial suite (must ship with tool)

- Negation flips (“not soluble”)  
- Number typos vs true change  
- Coreference (“it” → wrong entity if dropped)  
- Multi-fact sentences (partial extract)  
- Tool JSON field drops  
- Conflicting sources  

Any miss on hard suite = release blocker for “100% $M^\star$” label.

---

## 7. Failure modes (design against these)

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| **Extractor under-recall** | Silent claim loss | Multi-extractor union; span residual dump |
| **Extractor over-recall / hallucination** | Invented facts | Span grounding mandatory |
| **Canonicalization bugs** | False merge of different claims | Keep literals in key; conflict flag |
| **Fluency collapse** | Atomic-only → bad English | Mixed render + CE replay |
| **Tool policy wipe** | Knowledge cards only | Separate episode channel |
| **Distribution shift** | Val on original surface rises | Eval on both raw and compressed val |
| **Teacher cost** | LLM extract too slow/$$ | Cache; rule-first; batch offline |
| **Overclaiming “100%”** | Marketing lie | Versioned $M^\star$ only |

---

## 8. Relationship to existing LL3M pieces

| Existing | Role vs semcomp |
|----------|-----------------|
| `prepare_data.py` / story split | Stage 0–1 precursor |
| Exact + MinHash near-dedup | Stage 3 *document* layer; **keep**, then claim layer |
| Atomic synth + FineWeb short | Manual render profile ≈ `atomic` + filter |
| `ll3m/kb_facts.json` v2 | Approved external claim source + aliases |
| `tools.lookup` / Engram-lite | **Do not** force long-tail into train; compressor may **emit lookup cards** or leave to host |
| SciQ denser mixes (`cap_mix_v16_*`) | Ideal first customer of semcomp |
| Binary/ternary pack | Orthogonal (weight plane) |

**Principle:** semcomp increases **bits of unique teaching signal per token**. Quantization increases **params per MiB**. Engram increases **facts per host byte**. Together they raise system capability under L3; neither data nor weight plane replaces host memory for long-tail facts. Full argument: **§1b**. Also [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md).

---

## 9. Implementation plan (when built)

### Phase P0 — Spec + golden set (1–2 days)

- Freeze $M^\star$ schema v1  
- 200-doc golden corpus with hand claims  
- Coverage / invention unit tests  

### Phase P1 — Lossless-ish mechanical compressor (no LLM)

- Normalize + segment  
- Exact claim templates + tool episode parse  
- Doc + claim hash dedup  
- Atomic render + verify  
- CLI: `semcomp run --in ... --out ... --profile atomic`

**Target:** high $R_{\text{tok}}$ savings on boilerplate-heavy data; perfect verify on goldens.

### Phase P2 — Grounded LLM extract (optional teacher)

- Local or API teacher with span citations  
- Union with P1 rules  
- Cache by doc hash  
- Strict verify  

### Phase P3 — Train A/B on LL3M freeze shape

- A1 vs A2 under fixed steps  
- Promote profile if SciQ or agent improves without battery regression  

### Phase P4 — Product integration

- Hook into `prepare_data.py` / dense ladder Exp B  
- Manifest next to `mix_manifest.json`  
- Never default-on until verify gate green  

**Out of scope for v1:** multimodal, non-English, open-ended literary meaning, “compress weights.”

---

## 10. CLI sketch (non-normative)

```bash
# dry-run: report only
uv run python -m ll3m.semcomp estimate \
  --in data/raw/ --tokenizer data/tokenizer.json

# compress with hard guarantee
uv run python -m ll3m.semcomp run \
  --in data/raw/ \
  --out data/semcomp_v1/ \
  --profile mixed \
  --meaning-model hard_v1 \
  --verify strict \
  --refuse-on-residual

# audit
uv run python -m ll3m.semcomp audit \
  --corpus data/semcomp_v1/ \
  --against data/raw/ \
  --report data/semcomp_v1/coverage.json
```

Exit codes: `0` ok, `2` residual risk (no write if `--refuse-on-residual`), `3` invention detected.

---

## 11. Information-theoretic honesty box

- **Gzip / zstd:** ~lossless bits of *text*, no semantic ranking; train still pays full token tax after decode.  
- **Semantic compress:** lossy on *surface*, aimed lossless on $M^\star$; train pays only dense tokens.  
- **You cannot** preserve all conceivable meanings in fewer bits than the Kolmogorov complexity of those meanings.  
- **You can** preserve a **finite, typed claim set** and prove coverage. That is the scientific version of “100% of meaning.”

If a stakeholder needs “every stylistic nuance,” use a different profile (`identity` or light dedup only) and do not claim semantic compression.

---

## 12. Decision summary

| Question | Answer |
|----------|--------|
| Is 100% meaning possible? | **Yes for a versioned hard meaning model $M^\star$; no for unlimited interpretation** |
| What does the tool optimize? | Min tokens s.t. coverage($M^\star$)=100% and invention=0 |
| Core mechanism | Extract → claim dedup → deterministic render → **verify or refuse** |
| LL3M fit | Data-density lever under L3 param cap; complements Engram/tools |
| First win | SciQ / fact-heavy mixes and web boilerplate; mixed render for English |
| Ship bar | Golden suite + strict verify; train A/B non-regression |

---

## 13. Open research questions

1. Best **canonical key** for claims (string vs embedding vs symbolic triple)?  
2. How much **compact_prose** is required so base English does not collapse vs pure atomic?  
3. Can verify be **formal** (SMT on numbers/units) for a subset of claims?  
4. Should long-tail claims be **rerouted to Engram** instead of train surface (compress-by-offload)?  
5. Interaction with **binary 195M** train: does denser data close the SciQ gap faster than more params?

---

## 14. References

Numbered list for §3. Prefer arXiv IDs for permanence. Steal/skip ranking for new extractors also tracks in [ARXIV_SURVEY_LEARN_STEAL.md](ARXIV_SURVEY_LEARN_STEAL.md).

### Semantic / embedding deduplication

1. **SemDeDup** — Abbas et al. *SemDeDup: Data-efficient learning at web-scale through semantic deduplication.* arXiv:[2303.09540](https://arxiv.org/abs/2303.09540) (2023). Embedding-based removal of semantic near-duplicates; efficiency + quality on web-scale data. Code: [facebookresearch/SemDeDup](https://github.com/facebookresearch/SemDeDup/).  
2. **D4** — Tirumala et al. *D4: Improving LLM Pretraining via Document De-Duplication and Diversification.* arXiv:[2308.12284](https://arxiv.org/abs/2308.12284) (2023); NeurIPS Datasets & Benchmarks. De-dup + diversity selection in embedding space for LLM pretrain.  
3. **NeMo Curator semantic dedup** — NVIDIA. *Semantic Deduplication* in NeMo Curator / Framework docs (production pipeline: embed → cluster → pairwise → remove). [docs.nvidia.com/nemo/curator/…/semdedup](https://docs.nvidia.com/nemo/curator/curate-text/process-data/deduplication/semdedup).

### Quality filtering and dataset design testbeds

8. **FineWeb / FineWeb-Edu** — Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* arXiv:[2406.17557](https://arxiv.org/abs/2406.17557) (2024). Large-scale web curation; **FineWeb-Edu** educational filter improves knowledge/reasoning benches.  
9. **DataComp** — Gadre et al. *DataComp: In search of the next generation of multimodal datasets.* arXiv:[2304.14108](https://arxiv.org/abs/2304.14108) (2023). Fixed-model, vary-data benchmark (image–text); template for “curation is the experiment.”  
10. **DCLM (DataComp-LM)** — Li et al. *DataComp-LM: In search of the next generation of training sets for language models.* arXiv:[2406.11794](https://arxiv.org/abs/2406.11794) (2024). Controlled LM dataset experiments; filtering recipes matter at fixed train compute.

### Dense educational / synthetic surface

11. **Textbooks Are All You Need (phi-1)** — Gunasekar et al. arXiv:[2306.11644](https://arxiv.org/abs/2306.11644) (2023). Textbook-quality + synthetic data for small code LM.  
12. **Textbooks Are All You Need II (phi-1.5)** — Li et al. arXiv:[2309.05463](https://arxiv.org/abs/2309.05463) (2023). Synthetic textbook common sense / science for small general LM.  
13. **Cosmopedia** — Ben Allal et al. (Hugging Face). Synthetic textbooks dataset (2024). [huggingface.co/datasets/HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia).

### Selective training / example selection (orthogonal but related)

14. **Rho-1 / Selective Language Modeling** — Lin et al. *Rho-1: Not All Tokens Are What You Need.* arXiv:[2404.07965](https://arxiv.org/abs/2404.07965) (2024). Token-level useful/noisy split; train loss on high-value tokens only.  
15. **LESS** — Xia et al. *LESS: Selecting Influential Data for Targeted Instruction Tuning.* arXiv:[2402.04333](https://arxiv.org/abs/2402.04333) (2024). Influence-style selection; small SFT subsets can beat full data for a skill.

### Atomic claims and verification (meaning units)

16. **FActScore** — Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* arXiv:[2305.14251](https://arxiv.org/abs/2305.14251) (2023); EMNLP. Atomic fact decomposition + support rate.  
17. **Claimify** — Microsoft Research. *Towards Effective Extraction and Evaluation of Factual Claims* / Claimify blog (2025). LLM claim extraction with ambiguity handling. [microsoft.com/en-us/research/blog/claimify-…](https://www.microsoft.com/en-us/research/blog/claimify-extracting-high-quality-claims-from-language-model-outputs/).  
18. **AFEV** — Zheng et al. *Deconstructing Complex Claims via LLM-based Atomic Fact Extraction and Verification.* (2025). Atomic decomposition for fine-grained verification. arXiv HTML track e.g. [2506.07446](https://arxiv.org/html/2506.07446v1).  
19. **LLM-NERRE** — Dagdelen et al. *Structured information extraction from scientific text with large language models.* Nature Communications (2024). Fine-tune LLM for joint entity/relation → structured records. [doi:10.1038/s41467-024-45563-x](https://www.nature.com/articles/s41467-024-45563-x).

### Host memory / knowledge outside weights

20. **Engram** — DeepSeek. Conditional memory / knowledge lookup as first-class axis (arXiv:[2601.07372](https://arxiv.org/abs/2601.07372)). Facts outside residual stream; aligns with LL3M tools + KB.

### Project-local (this repo)

21. [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) — atomic synth + MinHash near-dedup + QAT ladder.  
22. [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md) — Phi/Smol/Qwen-small capacity math vs L3 pack.  
23. `ll3m/semcomp/` + `data/semcomp_tiny/` — P0/P1 scaffold and smoke.

### How our work cites them (quick matrix)

| Our stage | Closest prior art |
|-----------|-------------------|
| Stage 0 normalize / filter | FineWeb-Edu [8], DCLM [10] |
| Stage 3 doc near-dup | SemDeDup [1], D4 [2], MinHash |
| Stage 2–3 **claim** extract/dedup | FActScore [16], Claimify [17], AFEV [18], LLM-NERRE [19] |
| Stage 4 dense render | Phi textbooks [11][12], Cosmopedia [13] |
| Stage 5 coverage gate | FActScore-style support; stricter: no invention |
| Train-time alternative | Rho-1 [14], LESS [15] |
| Long-tail offload | Engram [20], our tools/KB |

---

## 15. Scaffold status (P0/P1 mechanical)

**Implemented (2026-08-10):**

| Path | Role |
|------|------|
| `ll3m/semcomp/` | package: schema, normalize, segment, extract, dedup, render, verify, pipeline, cli |
| `python -m ll3m.semcomp` / `ll3m-semcomp` | `estimate` \| `run` \| `audit` |
| `tests/fixtures/semcomp/golden_source.txt` | multi-doc golden |
| `tests/test_semcomp.py` | unit tests |

```bash
# estimate ratios
uv run python -m ll3m.semcomp estimate --in tests/fixtures/semcomp/golden_source.txt

# compress (writes mix.txt only if claim-list gate ok, unless --no-refuse-on-residual)
uv run python -m ll3m.semcomp run \
  --in tests/fixtures/semcomp/golden_source.txt \
  --out /tmp/semcomp_out --profile atomic --verify strict

# tiny real slice from cap_mix (40 blocks)
uv run python scripts/semcomp_tiny_slice.py --max-blocks 40

uv run pytest tests/test_semcomp.py -q
```

### Tiny cap_mix smoke (2026-08-10)

| Metric | Value |
|--------|--------|
| Source | 40 blocks from `cap_mix_v16` + `v15` (QA / tools; stories dropped) |
| Claims | **48** (deduped; 8 conflicts kept) |
| Claim-list gate | **ok** (missing=0 invented=0) |
| BPE tokens | slice **~2775** → mix **~1231** (**ratio ~0.44**) |
| Char ratio | ~0.71 (surface-only mix) |
| Round-trip re-extract | residual advisory (math Q/A keying); not hard-fail |

Artifacts: `data/semcomp_tiny/`.

**Still open:**

1. Expand golden to ~200 hand claims.  
2. Span-grounded LLM extract (P2); tighten round-trip for calc Q/A.  
3. Larger cap_mix / FineWeb slice.  
4. Train A/B on LL3M freeze shape.

Until P3 promotes a profile, keep using **manual atomic + MinHash** as the production densifier.
