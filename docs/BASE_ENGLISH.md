# Product language goal: proper base English (not stories)

**Snapshot:** 2026-07-20  
**Decision:** Prefer **general base English** (chat-ready fluency, clear prose, short Q&A) over TinyStories-style narrative as the primary capability.

**Related:** [PRODUCT_CANDIDATE.md](PRODUCT_CANDIDATE.md) · [KNOWLEDGE_DENSITY_SMALL_MODELS.md](KNOWLEDGE_DENSITY_SMALL_MODELS.md) · [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) · [NATIVE_TERNARY_PIPELINE.md](NATIVE_TERNARY_PIPELINE.md) · [TOOL_CALLING.md](TOOL_CALLING.md)

**Active denser rebuild (spark1):** see [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) (atomic-heavy mix, MinHash, KB v2, full QAT A→D, MC vs Qwen4).

---

## 0. Curriculum (locked order)

```text
Stage 1 — BASE (now / v76)
  Basic English + compressed knowledge density
  · FineWeb-Edu (edu web) + dense synthetic fact/explainer cards
  · Native ternary from step 0 (L3T2 grid)
  · Goal: fluent short English + high-signal common knowledge in weights
  · NOT tool transcripts yet

Stage 2 — TOOL SFT (after Stage 1 finishes)
  · When2Call mix: calc / lookup / time / no-call ~30–40%
  · Keep --qat ternary --embed-qat int8
  · Goal: when/how to emit <call>…</call>; host executes tools
  · Do not start until Stage 1 val + non-story samples look sane

Stage 3 — Serve
  · .ll3m + host tools (lookup _KB / wiki / calc / time)
  · Long-tail / live facts stay outside weights (Engram-lite)
```

| Do first | Do second | Never as Stage 1 |
|----------|-----------|------------------|
| Basic English | Tool SFT | Only multi-step tool distillations |
| Compressed KB density (edu + textbook-style cards) | Expand host `_KB` as needed | Pure TinyStories primary |
| Native ternary base train | Agent smoke with `--no-assist-tools` | Shakespeare-heavy mix |

**Compressed KB** here means **dense, short educational facts and explainers in the pretrain mix** (Phi-style), not stuffing Wikipedia into ternary and not skipping English for tool logs. Host `_KB` / wiki remain the open-ended fact store at serve time.

---

## 1. Why current models “sound like stories”

| Layer | Current | Effect |
|-------|---------|--------|
| Pretrain corpus | **TinyStoriesV2-GPT4** + 5% Shakespeare | Prior is children’s narrative |
| Tokenizer | BPE V=2048 trained on that mix | Tokens optimized for story words, weak for chat/news/tech |
| Eval / samples | “Once upon a time…” | Optimizes and demos story mode |
| Tool SFT | After story base | Can teach `<call>` tags; cannot invent adult English distribution |

Language eval (2026-07-20) confirmed: **story mode OK**, chat / facts / math **collapse** into Shakespeare fragments or story dumps. That is data, not architecture.

Serve freeze (L23×d576, L3T2, tools for facts) **stays**. **Data + tokenizer + train curriculum** change.

---

## 2. What “proper base English” means here

| In scope (weights) | Out of scope (host / later) |
|--------------------|-----------------------------|
| Grammatical, natural adult/simple English | Encyclopedic knowledge in ternary |
| Short answers, dialogue turns | Multi-page essays |
| Instructions & brief explanations | Long-context documents (T=256) |
| Enough fluency to wrap tool results | Live facts (wiki/lookup/calc) |
| When-not-to-call plain replies | Heavy multi-tool enterprise APIs |

Success looks like: *User: What time is it?* → call or clean refusal; *User: Explain gravity simply* → short coherent English; not *Once upon a time there was a boy named Tim…*

---

## 3. Recommended stack (order)

See **§0 Curriculum**. Short form:

```text
tokenizer on base-en mix
  → Stage 1 native ternary (English + compressed KB)
  → Stage 2 tool SFT (keep QAT)
  → C serve + host tools
```

**Do not** only overtrain more TinyStories. That deepens the story prior.

**Do not** replace Stage 1 with only multi-step tool distillations. English + dense KB first; tools second.

---

## 4. Data mix for base English (~79M, T=256)

Target: **educational / clean general English**, Phi/SmolLM-inspired, filtered for length and readability. Prefer quality over raw Common Crawl.

### 4.1 Suggested mixture (starting point)

| Slice | Share (tokens) | Sources (examples) | Role |
|-------|----------------|--------------------|------|
| **Edu web** | 40–50% | FineWeb-Edu (or similar educational filter) | Broad fluent English + light knowledge |
| **Synthetic textbooks / explainers** | 15–25% | Cosmopedia-style, short chapters, simple science | Clear expository prose |
| **Dialogue / instruct-ish** | 10–15% | Short cleaned UltraChat / OpenHermes-style **rewrites** to plain English; or synthetic User/Assistant | Chat register |
| **Code-lite** | 5–10% | Simple Python/comments only (optional) | Structure; keep small at V=2048 |
| **Story spice** | **0–5%** | TinyStories optional, not primary | Diversity only |
| **Shakespeare / archaic** | **0%** (or ≪1%) | Drop from default | Current mix causes Assistant → verse collapse |

Hard filters (same spirit as `prepare_data.py`):

- Length: fit **T=256** windows; drop huge docs or chunk to short paragraphs  
- Printable ratio, low repetition, sentence-ish ends  
- Prefer grade ~6–10 English (readable, not baby talk, not academic sludge)  
- Dedup + train/val leak control  

Token budget aspiration: **≥1–2B tokens seen** under native ternary if compute allows (current story run was ~0.8B seen over ~466M unique). Unique high-quality tokens matter more than 10 epochs of stories.

### 4.2 What not to use as primary

| Avoid as main mix | Why |
|-------------------|-----|
| Pure TinyStories | Story prior (current failure mode for “proper English”) |
| Raw unfiltered web | Noise; wastes tiny capacity |
| Full Wikipedia dump | Facts better on host; long entities hurt small BPE |
| Only tool transcripts | Over-calling; weak free English |
| Heavy Shakespeare | Verse mode on chat prompts |

### 4.3 Tokenizer

**Rebuild BPE on the new mix** before a serious base retrain.

| Option | Notes |
|--------|--------|
| **V=2048** (default freeze) | Safest for L3 embed cost; must retrain on general English sample |
| **V=4096** | Better word coverage; check `infer_ws` still ≤ 30.40 MiB |

Story-trained V=2048 will **mutilate** general English even with a better corpus.

---

## 5. Train recipe (native ternary, product grid)

Keep: `--qat ternary --embed-qat int8`, Lion, z-loss, MTP, LayerScale, softcap off, L23×d576.

| Stage | Data | Steps / LR (order) | Goal |
|-------|------|--------------------|------|
| **Base** | §4 mix | 50k–100k+ or until val plateaus; lr ~2–3e-4, warmup 3k | Base English under serve grid |
| **Instruct (optional)** | Short Q&A / chat, no tools | few k steps, lr ~5e-5 | Assistant register |
| **Tool SFT** | When2Call mix, ~30–40% no-call | few k steps, lr ~5e-5, QAT on | Tool policy |

Eval gates (replace story-only demos):

| Gate | Pass |
|------|------|
| Val NLL on **held-out general English** (not only TinyStories val) | Report + trend down |
| Prompts: openers, explain-X-simply, short dialogue | Coherent, non-story, non-verse |
| No “Once upon a time” default on neutral prompts | Qualitative |
| After tool SFT: calc/lookup + **no-call** chat | Protocol + plain English answers |
| L3 WS | Still ≤ 30.40 MiB |

---

## 6. Relation to existing checkpoints

| Checkpoint | Fit for new goal |
|------------|------------------|
| `v75` / natives / post-hoc | **Story-domain** bases; keep as ablations / history |
| Tool SFT on story native | Teaches tags on wrong prior; fine for protocol experiments, **not** final base English product |
| New run | Required for product language goal |

No amount of tool SFT alone turns Tim-and-Lily English into general base English.

---

## 7. Implementation checklist

1. [x] Freeze product language goal in this doc + `PRODUCT_CANDIDATE.md`  
2. [x] Scripts: `download_base_english.py`, `prepare_base_english.py` (no Shakespeare default)  
3. [x] Train / pipeline on **spark2 only**: `run_v76_base_en_pipeline_cuda.sh`, `train_v76_base_en_cuda.sh`  
4. [x] Launch from laptop: `launch_spark2_base_en.sh` (rsync + remote screen)  
5. [x] Local status mirror (optional, reboot-safe): `poll_spark2_base_en.sh` → `checkpoints/v76_base_en_status/`  
6. [x] Base English native ternary on spark (`checkpoints/v76_base_en/`, step ~60k, QAT ternary+int8)  
7. [x] Industry MC suite + Qwen3.5-0.8B NF4 baseline (`scripts/run_std_benchmarks.py`, `checkpoints/compare/`)  
8. [~] Denser ladder on spark1: KB v2 + atomic mix + continue QAT + tool SFT ([DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md))  
9. [ ] Language samples + final agent battery on `v76_dense_tool`  
10. [ ] Export product `.ll3m` after denser D2  

### Denser mix knobs (ladder defaults)

| Knob | Value |
|------|-------|
| Synthetic chars | 700M (kb_qa / fact / card heavy) |
| FineWeb-Edu chars | 500M (doc cap 1200) |
| Near-dedup Jaccard | 0.85 (MinHash) |
| Prepare max-chars | 1200 |
| Host KB | v2 ~471 entries / ~1164 keys |

### Reboot safety

| Component | Where | Survives local reboot? |
|-----------|--------|-------------------------|
| Data download + prepare + train | **spark2** GNU screen `ll3m-base-en` | **Yes** (only spark2 must stay up) |
| Local poller | optional on laptop | No (and must not matter) |
| Local status files | `checkpoints/v76_base_en_status/` | Yes (disk); refresh with poll script |

```bash
# start (from local, once)
bash scripts/launch_spark2_base_en.sh

# status anytime (optional)
bash scripts/poll_spark2_base_en.sh
bash scripts/poll_spark2_base_en.sh --watch

# attach on spark2
ssh -i ~/.config/NVIDIA/Sync/config/nvsync.key -o IdentitiesOnly=yes maci@192.168.0.212 -t screen -r ll3m-base-en
```

---

## 8. One-line summary

**Proper base English = change the pretrain distribution (and tokenizer), keep ternary L3 serve and host tools for facts; stories become optional spice, not the product.**
