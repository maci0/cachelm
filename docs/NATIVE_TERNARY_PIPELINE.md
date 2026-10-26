# Native ternary product pipeline (research-backed)

Maps **docs/RESEARCH_MODELS_2026.md** into a concrete train → serve → tools path
for L23×d576 MQA, L3T2 pack, host tools.

## Stance (unchanged)

| Layer | Holds |
|-------|--------|
| **Weights (ternary)** | Language + tool **policy** |
| **Host tools / KB** | Facts (Engram thesis) |
| **C serve** | L3T2 + MQA T=256 under ~30.4 MiB |

Architecture stays dense modern Transformer. No MoE / MLA / Mamba / diffusion.

---

## Stage map

| Stage | What | Research source | Script / entry |
|-------|------|-----------------|----------------|
| **0. Grid** | Native STE ternary Linears + int8 embeds from step 0 | BitNet b1.58, Tequila, TernaryLM, bitnet.cpp I2_S | `scripts/train_v75_native_ternary.sh` |
| **1. Train objectives** | CE + z-loss + shared-head MTP + LayerScale | DeepSeek MTP, PaLM z-loss, deep residual | flags in train script |
| **2. Stability** | Longer warmup (3k), grad clip 1.0, Lion, dropout 0.1 | BitNet small-net care; Kimi/Gemma softcap optional | script defaults |
| **3. Export** | L3T2 `.ll3m` same absmean pack as STE | bitnet.cpp train–serve lock | `export_weights.py --quant ternary` |
| **4. Compare** | Native vs fp32 vs post-hoc QAT on serve grid | BitNet quality story | `compare_fp32_vs_qat.py` |
| **5. Base English + compressed KB** | Edu web + dense fact/explainer cards (not stories-primary) | Phi / FineWeb-Edu / BASE_ENGLISH.md | `v76_base_en` |
| **5b. Denser ladder** | Atomic synth + short FineWeb + MinHash + continue QAT + fact/tool SFT | [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) | spark1 `v76_dense_*` |
| **6. Tool SFT** | Gold call + no-call ~1:1; keep QAT on; **after** base English | Kimi agentic data, BFCL/When2Call | `gen_tool_sft.py` + `train_tool_sft.sh` (also ladder D2) |
| **7. Host knowledge** | KB v2 entries+aliases, word-bound lookup, wiki/calc | Engram O(1) external memory; GLM5.2 retrieval-lift | `ll3m/kb_facts.json` + `tools.py` |
| **7b. Agent loop** | Short `<call>` protocol; `gate_tools`; assist bootstrap until SFT sticks | Kimi tool template; assist → `--no-assist` after SFT | `ll3m/agent.py` chat `--mode agent` |
| **8. Later (optional)** | KD from 0.8B–3B teacher; TL GEMV; constrained decode | Ministral cascade, Elastic, bitnet.cpp, SimpleTool | backlog R-C, R-E, bn7 |

### Explicitly **not** in this pipeline

| Idea | Why skip |
|------|----------|
| Softcap default ON | C runtime has no softcap; train/serve logit path would diverge |
| MoE / Mamba / MLA / DSA | T=256 L3 residency; wrong axis |
| Diffusion decode | Different C graph |
| Pack world knowledge into ternary | Density fails; Engram/tools (see `docs/KNOWLEDGE_DENSITY_SMALL_MODELS.md`) |

---

## Default hyperparameters (stage 0)

| Knob | Value | Why |
|------|-------|-----|
| Shape | L23 d576 h12/kv1 mid1536 | Product freeze |
| `--qat ternary` | from step 0 | Native BitNet grid |
| `--embed-qat int8` | from step 0 | Match L3T2 embeds |
| steps | 50000 | Match v75 token budget class |
| lr | 3e-4 cosine | Same order as fp32 phase-1 |
| warmup | 3000 | STE noisier at init |
| dropout | 0.1 | Small-net regularize |
| layer-scale | 1e-4 | Deep residual (no peri-LN: failed at this width) |
| z-loss | 1e-4 | Logit stability |
| mtp | 0.1 / depth 2 | Multi-token train signal (serve free) |
| logit-softcap | **0 (off)** | C parity |
| optimizer | lion | Existing default |

Override via env: `STEPS=60000 LR=2e-4 SOFTCAP=30 bash scripts/train_v75_native_ternary.sh`.

---

## Stage 5 tool mix (When2Call)

Target episode mix in `gen_tool_sft.py`:

| Type | ~Share | Goal |
|------|--------|------|
| calc | 30% | numeric tools |
| lookup | 20% | closed KB / Engram-lite |
| time | 10% | host clock |
| **no-call** | **35%** | when **not** to call |
| error recovery | 5% | bad args → retry |

After native best.pt: `CKPT=checkpoints/v75_native_ternary/best.pt bash scripts/train_tool_sft.sh`  
**Must keep** `--qat ternary --embed-qat int8` so serve grid does not drift.

---

## Automation

### Preferred trainer: spark2 (NVIDIA GB10 @ 192.168.0.212)

```text
spark2: scripts/train_v75_native_ternary_cuda.sh  (device=cuda, --force for remote L3 size)
    → ~/Desktop/cachelm/checkpoints/v75_native_ternary/ on spark2
local ROCm: checkpoints/v75_native_ternary/   (separate run; never overwritten by pull)
local pull: scripts/pull_spark2_native_when_ready.sh
    → rsync → checkpoints/v75_native_ternary_spark2/
    → export best.ll3m + quality_compare vs v75 / v75_ternary / local native
```

SSH key: `~/.config/NVIDIA/Sync/config/nvsync.key` (host `spark2` in NVIDIA Sync config).

`--force` is required on spark2: GB10 reports ~8 MiB L3; product L3 gate (30.4 MiB) applies to the 9950X **serve** machine only.

### Local AMD (ROCm) path

```text
v75 post-hoc QAT finish
    → scripts/run_v75_native_ternary_when_ready.sh
         → train native on ROCm
         → export + compare (+ optional tool SFT)
```

Manual:

```bash
# remote (preferred)
ssh -i ~/.config/NVIDIA/Sync/config/nvsync.key -o IdentitiesOnly=yes maci@192.168.0.212 \
  'cd ~/Desktop/cachelm && nohup bash scripts/train_v75_native_ternary_cuda.sh > checkpoints/v75_native_ternary/train.log 2>&1 &'

# local ROCm
bash scripts/train_v75_native_ternary.sh

# after weights home:
CKPT=checkpoints/v75_native_ternary/best.pt OUT=checkpoints/v75_native_tool \
  bash scripts/train_tool_sft.sh
```

---

## Success metrics

| Check | Pass |
|-------|------|
| Infer WS | ≤ 30.40 MiB ternary+int8 |
| Val CE (native) | Report vs v75 1.10 and v75_ternary ~1.22 |
| Serve-grid Δ | `quality_compare*.json` |
| Agent | calc/lookup/time with `--no-assist-tools` after tool SFT |
| Residency | hard CCD gate still PASS |

---

## Colibrì-inspired serve (implemented)

| Item | Where |
|------|--------|
| Clip babble after `</call>`; repair incomplete open tags | `ll3m/constrained.py` → `agent` |
| Quant-aware sample defaults (story 0.7/30, agent 0.4/20) | `quant_sample_defaults`, chat/generate CLI |
| Quality > silent fit | L3 gate + `--force` only when intentional (e.g. spark2) |

## Backlog still open (after native baseline)

See `AGENTS.md` R-B…R-J and quant 3–7: train–serve BitLinear lock, TL GEMV A/B on decode shape, KD, host Engram hash tables beyond static `_KB`, optional C-side token mask for tools.
