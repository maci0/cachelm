# LL3M status and training pipeline

**Snapshot:** 2026-07-20 (base-English launch)  
**Product:** dense L23×d576 MQA (~78.8M), ternary L3T2 serve, L3 CCD gate **30.40 MiB** (9950X serve claim)  
**Language goal:** **Stage 1** basic English + compressed KB → **Stage 2** tool SFT (not before). See `docs/BASE_ENGLISH.md` §0.

---

## 1. Live jobs (this snapshot)

| Job | Host | Status | Notes |
|-----|------|--------|--------|
| Phase-1 fp32 (`v75`) | local | **done** 50k | story-domain · best_val **1.100** |
| Post-hoc QAT (`v75_ternary`) | local ROCm | **done** 53k | story-domain |
| Native ternary local | local ROCm | **done** 50k | story-domain · serve CE **1.206** |
| Native ternary spark2 | GB10 | **done** 50k | story-domain |
| Tool SFT on story native | local | may finish soon | not final product base |
| **v76 base-English Stage 1** | spark2 → local | **done** 60k | best_val ~**2.52** · `best.ll3m` · C tok fixed |
| **v76 tool SFT Stage 2** | spark2 → local | **done** +3k | best_val **0.086** · `v76_base_en_tool/best.ll3m` · agent calc smoke OK |

### Reboot note (critical)

| Process | Machine | Survives **local** reboot? |
|---------|---------|----------------------------|
| Data + train pipeline | **spark2** `screen ll3m-base-en` | **Yes** |
| Local status poller | this laptop | **No** (restart with poll script) |

```bash
bash scripts/poll_spark2_base_en.sh           # once
bash scripts/poll_spark2_base_en.sh --watch   # loop
# mirror: checkpoints/v76_base_en_status/latest.txt
# attach: ssh … -t screen -r ll3m-base-en
```

Do **not** rely on any local waiter for train. After local reboot, only re-run the poller if you want a live mirror.

### Hosts

| Name | Address | Role | SSH |
|------|---------|------|-----|
| local | this machine (9950X + 7900 XTX ROCm) | serve claim, QAT, tool SFT, export | n/a |
| **spark2** | `maci@192.168.0.212` | native ternary train (CUDA, finished) | NVIDIA Sync key |

```bash
ssh -i ~/.config/NVIDIA/Sync/config/nvsync.key -o IdentitiesOnly=yes maci@192.168.0.212
screen -r ll3m-tool-sft   # local tool SFT
```

---

## 2. Scoreboard (finished baselines)

### Train val CE (`progress.json` best_val)

| Run | best_val | Pack |
|-----|----------|------|
| v75 fp32 | **1.100** | `checkpoints/v75/best.{pt,ll3m}` |
| v75_ternary post-hoc QAT | **1.218** | `checkpoints/v75_ternary/best.{pt,ll3m}` |
| **local native** (ROCm) | **1.172** | `checkpoints/v75_native_ternary/best.{pt,ll3m}` |
| spark2 native (CUDA) | **1.170** | `checkpoints/v75_native_ternary_spark2/best.{pt,ll3m}` |

### Serve-grid CE (forced ternary Linear + int8 emb = C path)

| Pack | serve-grid CE | Notes |
|------|---------------|--------|
| fp32 forced onto quant grid | 2.275 | collapses without QAT |
| post-hoc QAT | 1.260 | large win vs naive quant |
| **local native** | **1.206** | **best quantized** |
| spark2 native | 1.213 | +0.007 vs local; still beats post-hoc by ~0.047 |

Smoke story (`complete`, T=0.7/top_k=30): both native `.ll3m` packs produce coherent TinyStories-style text.

**Winner for tool SFT / product serve:** local `v75_native_ternary` (serve-grid). spark2 kept as alternate pack.

---

## 3. Checkpoint tree

| Path | What | Status |
|------|------|--------|
| `checkpoints/v75/` | Phase-1 fp32 | done |
| `checkpoints/v75_ternary/` | Post-hoc QAT + compare | done |
| `checkpoints/v75_native_ternary/` | Local ROCm native | done · **export done** `best.ll3m` |
| `checkpoints/v75_native_ternary_spark2/` | spark2 pull | done · compares vs fp32/posthoc/local |
| `checkpoints/v75_native_tool/` | Tool SFT on local native | **in progress** (keep qat=ternary embed_qat=int8) |

Local native tree is never overwritten by pull.

---

## 4. Product freeze (serve)

| Field | Value |
|-------|--------|
| Shape | L23 × d576, heads 12/1 (MQA), mid 1536, T=256, V=2048 BPE |
| Params | 78,815,712 |
| Serve pack | **L3T2**: ternary Linears (2-bit) + int8 embeds |
| Infer WS | ~**30.22 MiB** / gate **30.40 MiB** (FITS on 9950X CCD) |
| C engine | `c_infer/ll3m_c`, Python wrap default |
| Knowledge | **host tools** only: calc / lookup / wiki / time / note · Engram-lite `_KB` in `ll3m/tools.py` |
| **Language goal** | **Base English** (not TinyStories-primary). Current `v75*` packs are story-skewed. Plan: `docs/BASE_ENGLISH.md` |

L3 claim is **serve-only** on the 9950X. Training may use full GPU/DRAM.

---

## 5. Data and knowledge stance

**Do not pack encyclopedia / general world knowledge into the ~21 MiB ternary weights.** Density fails; research (Engram, GLM5.2 retrieval-lift) says externalize facts.

| Layer | Dataset / store | Role |
|-------|-----------------|------|
| **Pretrain language** | **TinyStoriesV2-GPT4** (~1.5 GiB partial → ~466M train tokens) + 5% TinyShakespeare spice | English fluency, narrative grammar |
| **Tokenizer** | domain BPE V=2048 on story mix (`data/tokenizer.json`) | fit short context |
| **Tool policy SFT** | synthetic When2Call-style mix (`scripts/gen_tool_sft.py` → `data/tool_sft.txt`) | when/how to emit `<call>` |
| **Facts at serve** | host `lookup` closed KB + `wiki` + `calc` + `time` | general knowledge **outside** weights |

Optional later (not required for product thesis): short Q&A curriculum, public tool traces rewritten to our tags (xLAM / Glaive / When2Call), or KD from a larger teacher. **Not** Wikipedia dump into weights.

Full map: `docs/NATIVE_TERNARY_PIPELINE.md` · survey: `docs/RESEARCH_MODELS_2026.md` · knowledge density: `docs/KNOWLEDGE_DENSITY_SMALL_MODELS.md` · tools: `docs/TOOL_CALLING.md`

---

## 6. Training pipeline

```text
  TinyStoriesV2-GPT4 + Shakespeare
           │
           ▼
  A. Two-phase baseline (done)
     v75 fp32 50k → v75_ternary post-hoc QAT
           │
  B. Native ternary preferred (done)
     local ROCm + spark2 CUDA, step 0 STE
     serve-grid compares → local native wins CE
           │
  C. Tool SFT (running) keep QAT on
     gen_tool_sft → train_tool_sft → v75_native_tool/best.ll3m
           │
           ▼
  C serve: best.ll3m + agent tools (knowledge on host)
```

| Stage | Script |
|-------|--------|
| Native train | `scripts/train_v75_native_ternary.sh` / `_cuda.sh` |
| Export | `scripts/export_weights.py --quant ternary` |
| Compare | `scripts/compare_fp32_vs_qat.py` |
| Tool SFT | `CKPT=…/best.pt OUT=checkpoints/v75_native_tool bash scripts/train_tool_sft.sh` |
| Agent | `python -m ll3m.chat --checkpoint …/best.ll3m chat --mode agent --no-assist-tools` (after SFT) |

---

## 7. How to monitor tool SFT

```bash
screen -ls
screen -r ll3m-tool-sft          # Ctrl-A d detach
cat checkpoints/v75_native_tool/progress.json
tail -f checkpoints/v75_native_tool/train_tool_sft.log
```

Rough ETA: ~3000 steps @ ~8k tok/s → **~1–1.5 h** wall (order of magnitude).

---

## 8. ROCm / CUDA notes

| Machine | Torch | Device |
|---------|-------|--------|
| local 7900 XTX | `2.10.0+rocm7.0` | `--device rocm` |
| spark2 GB10 | `2.13.0+cu130` | `--device cuda` |

**Footgun:** `uv sync` can replace ROCm torch with a CUDA wheel on AMD. Restore:

```bash
uv pip install --python .venv/bin/python 'torch==2.10.0+rocm7.0' \
  --index-url https://download.pytorch.org/whl/rocm7.0
```

---

## 9. Next after tool SFT

1. Confirm `checkpoints/v75_native_tool/best.ll3m` export  
2. Agent smoke with `--no-assist-tools` (calc / lookup / time)  
3. Optional: expand host `_KB` or tool mix; constrained call path already in `ll3m/constrained.py`  
4. Backlog: KD, TL GEMM A/B, BitLinear train–serve lock (`AGENTS.md`)

---

## 10. Key docs

| Doc | Role |
|-----|------|
| `status.md` | **This file** — live snapshot |
| `docs/NATIVE_TERNARY_PIPELINE.md` | Research → stages |
| `docs/RESEARCH_MODELS_2026.md` | Model/arxiv survey |
| `docs/KNOWLEDGE_DENSITY_SMALL_MODELS.md` | Small models + knowledge compression; LL3M mapping |
| `docs/PRODUCT_CANDIDATE.md` | Freeze + recipes |
| `docs/TOOL_CALLING.md` | Agent protocol + public tool datasets |
| `docs/QUANTIZATION.md` | Ternary / QAT / L3T2 |
| `AGENTS.md` | Ops + backlog |
