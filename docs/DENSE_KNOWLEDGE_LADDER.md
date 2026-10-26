# Dense knowledge ladder (v76 denser rebuild)

**Snapshot:** 2026-07-21  
**Host:** spark1 `192.168.0.211` (CUDA), screen `ll3m-dense`  
**Goal:** more **semantic knowledge per token and per host byte**, keep **ternary + int8 QAT** end-to-end, validate with **industry MC benches** vs **Qwen3.5-0.8B-Base NF4**.

**Related:** [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md) · [BASE_ENGLISH.md](BASE_ENGLISH.md) · [TOOL_CALLING.md](TOOL_CALLING.md) · [QUANTIZATION.md](QUANTIZATION.md) · [NATIVE_TERNARY_PIPELINE.md](NATIVE_TERNARY_PIPELINE.md)

---

## 1. Product split (non-negotiable)

| Layer | Holds | Location |
|-------|--------|----------|
| **Weights (L3T2)** | Language + tool **policy** under ternary Linears + int8 embeds | `.ll3m` / train QAT |
| **Host Engram-lite** | Canonical facts (lookup) | `ll3m/kb_facts.json` + `tools.lookup` |
| **Host open world** | Live / long-tail | `wiki` (capped extract), `calc`, `time` |
| **C serve** | Decode WS ≤ one L3 CCD (~30.4 MiB gate) | `c_infer/` |

Do **not** pack an encyclopedia into ternary. Density math and Engram thesis: [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md).

**QAT lock:** every train stage on this ladder uses `--qat ternary --embed-qat int8 --serve-quant ternary`. Pipeline refuses `QAT=none` and **asserts** checkpoint tags after base / C / D1 / D2.

---

## 2. Ladder stages (A → D)

| Exp | Name | What | Artifact |
|-----|------|------|----------|
| **A** | Host KB | Expand + normalize Engram-lite table; agent lookup smoke | `ll3m/kb_facts.json`, `tools.py` |
| **B** | Dense data | Atomic-heavy synthetic + shorter FineWeb + exact + **MinHash near-dedup** + BPE | `data/base_en_dense/` |
| **C** | Continue pretrain | Resume `v76_base_en` + **30k** steps on dense mix, QAT on | `checkpoints/v76_dense_c/` |
| **D1** | Fact Q&A SFT | Closed-book cards from KB flat map, QAT on | `checkpoints/v76_dense_qa/` |
| **D2** | Tool SFT | When2Call mix, QAT on | `checkpoints/v76_dense_tool/` |
| **Validate** | Std MC vs Qwen4 | After C and after D | `vs_qwen4_expC.json`, `vs_qwen4_expD.json` |

### Defaults (denser rebuild, 2026-07-21)

| Knob | Value | Rationale |
|------|-------|-----------|
| `SYN_CHARS` | **700M** | Atomic fact/Q&A cards (high info/token) |
| `FINEWEB_CHARS` | **500M** | Edu language glue (shorter chunks) |
| FineWeb `max_doc` | **1200** chars | Cap long pages |
| Prepare `--max-chars` | **1200** | Same |
| Near-dedup Jaccard | **0.85** | Collapse near-copies |
| Prefer short weight | **1.6** | Bias train order toward short docs |
| `CONT_STEPS` | 30000 | Continue from base step ~60000 → ~90000 |
| `QA_STEPS` / `TOOL_STEPS` | 3000 each | Short SFT polish |
| `BENCH_LIMIT` | 400 | MC log-likelihood protocol |
| QAT | ternary + embed int8 | Product pack |

Earlier aborted run used SYN=400M / FW=1.2B without near-dedup; killed mid Exp C for this denser rebuild.

---

## 3. Experiment A: host KB v2

### Schema (`ll3m/kb_facts.json`)

```json
{
  "version": 2,
  "note": "Canonical Engram-lite KB...",
  "n_entries": 471,
  "n_keys": 1164,
  "entries": [
    {
      "id": "capital france",
      "value": "Paris",
      "aliases": ["capital of france", "france capital", "..."]
    }
  ],
  "flat": { "capital france": "Paris", "capital of france": "Paris", "...": "..." }
}
```

| Field | Role |
|-------|------|
| `entries[].id` | Canonical key |
| `entries[].value` | Short fact string (prefer tiny values) |
| `entries[].aliases` | Alternate surface forms → same value |
| `flat` | Pre-expanded map for loaders / train generators |

**Scale (this rebuild):** ~**471** canonical entries, ~**1164** lookup keys (was ~413 flat keys).

### Runtime (`ll3m/tools.py`)

- Loads v2 `flat` (or expands `entries`); still accepts v1 flat dict.
- `tool_lookup`: normalize query, strip leading “what is / define”, exact hit, else **longest word-bounded** key match.
- No bare substring match (`pi` must not hit inside `pizza`).

### Agent (`ll3m/agent.py`)

- Already used word-bounded closed-KB keys for gold tool assist.
- Tool SFT (`gen_tool_sft.py`) samples keys from `_KB` (all flat aliases available).

### Ops

```bash
# smoke
.venv/bin/python -c "from ll3m.tools import _KB, tool_lookup; print(len(_KB), tool_lookup('capital of france'))"
# expect: 1164+ and Paris
```

---

## 4. Experiment B: denser pretrain mix

### Download (`scripts/download_base_english.py`)

| Source | Role |
|--------|------|
| **Synthetic** | KB Q&A / one-line facts / knowledge cards + light explainers / math / prose |
| **FineWeb-Edu** | Streaming `sample-10BT`, edu `score ≥ 1.5`, doc cap **1200** chars |
| Stories | Default **0** |

Synthetic kind weights (density bias):

| Kind | Weight | Form |
|------|--------|------|
| kb_qa | 34 | `User: …\nAssistant: …` |
| kb_fact | 24 | `Fact: k = v` |
| kb_card | 18 | Short topic + fact |
| explainer / dialogue / math / prose | lower | English glue only |

KB for synth is the v2 **flat** map (all aliases as keys → more surface forms in pretrain).

### Prepare (`scripts/prepare_base_english.py`)

Pipeline:

1. Split raw `*.txt` on blank lines → docs  
2. Quality filter (length, printable, word-repeat, sentence-ish end)  
3. **Exact dedup** (whitespace-normalized SHA1)  
4. **MinHash near-dedup** (word 3-grams, 32 perms, 8 bands, Jaccard ≥ 0.85)  
5. Short-doc weighted shuffle (`prefer-short-weight`)  
6. Train/val split → `mix_*.txt` + BPE `tokenizer.json` + `mix_*.bin` + `manifest.json`

```bash
uv run python scripts/download_base_english.py \
  --out-dir data/base_en_dense/raw \
  --fineweb-chars 500000000 \
  --synthetic-chars 700000000 \
  --stories-chars 0

uv run python scripts/prepare_base_english.py \
  --raw-dir data/base_en_dense/raw \
  --out-dir data/base_en_dense \
  --vocab-size 2048 \
  --max-chars 1200 \
  --near-dedup-jaccard 0.85 \
  --prefer-short-weight 1.6
```

---

## 5. Experiments C / D: full QAT train

Resume always from a **ternary + int8** checkpoint (base `v76_base_en/best.pt` at step ~60000 is already tagged).

| Stage | Out-dir | Data | Steps (delta) | QAT |
|-------|---------|------|---------------|-----|
| C | `checkpoints/v76_dense_c` | `data/base_en_dense/mix_{train,val}.txt` | +30000 | ternary + int8 |
| D1 | `checkpoints/v76_dense_qa` | `data/fact_qa_sft.txt` (from KB) | +3000 | same |
| D2 | `checkpoints/v76_dense_tool` | `data/tool_sft.txt` (When2Call) | +3000 | same |

Export after C and D: `scripts/export_weights.py --quant ternary` → `best.ll3m`.

**Assert:** each finished `best.pt` must have `qat=="ternary"` and `embed_qat=="int8"` or the pipeline exits.

---

## 6. Validation protocol (industry MC + Qwen baseline)

### Benchmarks (`scripts/run_std_benchmarks.py`)

Zero-shot **continuation log-likelihood** multiple choice, `limit=400`, `seed=42`:

| Task | Notes |
|------|--------|
| HellaSwag | 4-way |
| ARC-Easy | science |
| WinoGrande | 2-way |
| BoolQ | 2-way |
| OpenBookQA | 4-way |
| SciQ | 4-way |
| COPA | 2-way (n=100 full set if smaller) |

### Compare (`scripts/validate_vs_qwen4.py`)

1. Run MC on the LL3M `.pt`  
2. Diff vs frozen baseline:

`checkpoints/compare/Qwen3.5-0.8B-Base_int4_limit400.json`

(Qwen3.5-0.8B-Base, bitsandbytes **NF4**, same limit/seed protocol.)

Prior reference (story/tool path, not denser ladder):

| Model | Params (approx) | Macro acc |
|-------|-----------------|-----------|
| LL3M `v76_base_en_tool` | ~79M ternary pack ~21 MiB | ~**40.1%** |
| Qwen3.5-0.8B-Base NF4 | ~504M | ~**60.2%** |

Expect denser ladder to move LL3M macro; **not** Qwen parity at 25× fewer params / L3 pack. Product claim remains agent + English under L3, not MMLU race.

Outputs per stage:

| File | Meaning |
|------|---------|
| `std_benchmarks_expC_limit400.json` | Raw MC |
| `vs_qwen4_expC.json` | Per-task deltas vs Qwen |
| same for `expD` under tool out-dir | Final |

---

## 7. Ops: spark1 pipeline

### Scripts

| Script | Role |
|--------|------|
| `scripts/launch_spark1_dense.sh` | Local → rsync code + Qwen baseline JSON + start screen |
| `scripts/run_spark1_dense_pipeline.sh` | A→D + QAT lock + benches (INNER=1 inside screen) |
| `scripts/poll_spark1_dense.sh` | Optional local status mirror (`--watch`) |
| `scripts/kill_spark1_dense.sh` | Stop screen/train; clear dense ckpt + `data/base_en_dense` |
| `scripts/train_tool_sft.sh` | D2; refuses `QAT=none` |
| `scripts/gen_fact_qa_sft.py` | D1 data from KB flat |
| `scripts/gen_tool_sft.py` | D2 When2Call mix |

### SSH

```bash
KEY=~/.config/NVIDIA/Sync/config/nvsync.key
HOST=maci@192.168.0.211

# launch (from repo root on laptop)
bash scripts/launch_spark1_dense.sh

# status
bash scripts/poll_spark1_dense.sh
ssh -i $KEY -o IdentitiesOnly=yes $HOST \
  'tail -f ~/Desktop/cachelm/checkpoints/dense_pipeline/pipeline.log'

# attach
ssh -i $KEY -o IdentitiesOnly=yes $HOST -t screen -r ll3m-dense

# kill + wipe dense tree (keeps v76_base_en)
# (run on spark1)
bash scripts/kill_spark1_dense.sh
```

Remote root: `/home/maci/Desktop/cachelm`  
Log: `checkpoints/dense_pipeline/pipeline.log`  
Done marker: `checkpoints/dense_pipeline/DONE.txt`

### Env overrides

| Env | Default | Meaning |
|-----|---------|---------|
| `SYN_CHARS` / `FINEWEB_CHARS` | 700M / 500M | Raw char budgets |
| `CONT_STEPS` | 30000 | Exp C delta steps |
| `QA_STEPS` / `TOOL_STEPS` | 3000 | D1 / D2 |
| `QAT` / `EMBED_QAT` / `SERVE_QUANT` | ternary / int8 / ternary | Locked product |
| `PREP_MAX_CHARS` | 1200 | Prepare doc cap |
| `NEAR_DEDUP_J` | 0.85 | 0 disables near-dedup |
| `SKIP_A` … `SKIP_D` | 0 | Skip stage if `1` |
| `BASE_CKPT` | `checkpoints/v76_base_en/best.pt` | Resume root |

Local reboot is **safe** once the screen is up on spark1.

---

## 8. Compression menu: done vs skipped

| Practice | Status |
|----------|--------|
| Host KV / Engram-lite | **Done** (v2 + aliases) |
| Word-boundary lookup | **Done** |
| Short wiki extract | **Done** (cap ~400) |
| Atomic synthetic cards | **Done** (700M, high kb_* weights) |
| Edu web filter (FineWeb-Edu score) | **Done** |
| Exact dedup | **Done** |
| MinHash near-dedup | **Done** |
| Short doc bias + 1200 cap | **Done** |
| QAT ternary+int8 all stages | **Done** (asserted) |
| Fact Q&A + tool SFT | **In pipeline** D1/D2 |
| Industry MC vs Qwen NF4 | **In pipeline** after C/D |
| Teacher rewrite of full FineWeb | **Skipped** (cost; optional later) |
| Vector embed RAG index | **Skipped** (wiki + lookup enough for now) |
| Triple store `(s,p,o)` | **Partial** (flat keys; schema is id+aliases not full RDF) |
| Bigger vocab for better BPE rate | **Rejected** (L3 embed budget) |

---

## 9. Checkpoint lineage

```text
v76_base_en/best.pt          # base English native ternary (spark2/spark1)
        │  denser mix + continue QAT
        ▼
v76_dense_c/best.pt          # Exp C
        │  fact Q&A SFT QAT
        ▼
v76_dense_qa/best.pt         # Exp D1
        │  tool SFT QAT
        ▼
v76_dense_tool/best.pt       # Exp D2 product candidate for agent + English
        │  export ternary
        ▼
v76_dense_tool/best.ll3m     # L3T2 serve
```

| Keep | Do not overwrite casually |
|------|---------------------------|
| `v76_base_en/` | Base before denser ladder |
| `v76_base_en_tool/` | Prior tool path (benches baseline) |
| `checkpoints/compare/*qwen*` | Frozen Qwen NF4 baseline JSON |

---

## 9c. Multi-mix SFT follow-up (after A–D)

Tool-only SFT (Exp D) flattened MC (SciQ drop). Next experiment:

**Multi-objective SFT from `v76_dense_c`:** dense replay (~50%) + fact Q&A (~20%) + tool episodes (~30%), keep ternary+int8 QAT, +5k steps, then MC vs Qwen4 + agent battery on C / D / MIX.

| Script | Role |
|--------|------|
| `scripts/build_multi_mix_sft.py` | Build `data/multi_mix_sft.txt` |
| `scripts/run_spark1_multi_mix_sft.sh` | Train → export → benches → batteries |
| Out | `checkpoints/v76_dense_mix/` |

```bash
# on spark1 (or launch screen)
bash scripts/run_spark1_multi_mix_sft.sh
# screen: ll3m-mix   log: checkpoints/dense_pipeline/multi_mix.log
```

---

## 9b. Recovery: missing `best.pt` after continue

If resume keeps a stale `best_val` from the base ckpt (same QAT, new data mix), val may never beat that bar and train writes only `last.pt`.

**Code mitigations:**
- `ll3m/train.py`: end of run writes `best.pt` from final weights if missing
- Pipeline `ensure_best_pt`: promotes `last.pt` → `best.pt` before QAT assert

**Ops (spark1):**
```bash
cp checkpoints/v76_dense_c/last.pt checkpoints/v76_dense_c/best.pt
# from laptop: reuse data + C weights → export, MC vs Qwen4, then D
SKIP_A=1 SKIP_B=1 bash scripts/launch_spark1_dense.sh
```

---

## 10. What success looks like

1. **A:** `KB_KEYS ≥ 800`, lookup aliases work, no pi⊂pizza.  
2. **B:** `manifest.json` with near_dedup stats, shorter median doc than old 2.9k-char mix.
   Local snapshot: `data/base_en_dense/manifest.json` records **161,074,628** train tokens
   (from `checkpoints/dense_pipeline/pipeline.log` ExpB_OK). Mix bins may remain only on
   the train host; do **not** cite `data/mix_manifest.json` (TinyStories ~466M) as denser-C.  
3. **C:** train finishes QAT tags intact; `vs_qwen4_expC.json` written.  
4. **D:** tool battery + `vs_qwen4_expD.json`; agent calc/lookup/no-call still gated.  
5. **Serve:** `best.ll3m` still under product L3 WS gate (shape unchanged).

Macro MC may remain well below Qwen0.8B; judge on **delta vs prior LL3M**, agent reliability, and residency, not frontier MMLU.

---

## 11. One-line summary

**Dense ladder = host-canonical facts + atomic-heavy deduped pretrain + short SFT, with ternary/int8 QAT never dropped, measured on standard MC against Qwen3.5-0.8B NF4 after each train stage.**
