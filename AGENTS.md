# AGENTS.md — LL3M

Instructions for coding agents working in this repository.

## Product goal (non-negotiable)

**Inference** working set (weights + activations + KV) must fit **one CPU L3 CCD**
and stay resident on the hot path (no DRAM round-trips for that data after pin/warm).

| Gate | Value on this host |
|------|---------------------|
| L3 per CCD | **32 MiB** (Ryzen 9 9950X: 2×32 MiB) |
| Product budget | **95% CCD ≈ 30.4 MiB** |
| Estimator | `estimate_infer_ws()` in `ll3m/model_torch.py` |
| Serve claim | Bare-metal **C** in `c_infer/`, not Python RSS |

**Training may use full DRAM / ROCm VRAM.** Only C serve carries the L3 claim.

Mental model: train big → export `.ll3m` → C decode WS ≤ one CCD.

## Product shape + continuous capability (2026-08-01)

**Shape freeze is lifted.** Retrain, finetune, RL, data, Engram, and shape search are all allowed.
Prefer **new out-dirs**; promote ship by metrics (no loyalty to pure_v2).
Default L3 goal still applies: infer WS ≤ one CCD (~30.4 MiB gate) unless product claim changes.
Loop + queue: `docs/CAPABILITY_OPEN.md`. Eval helper: `scripts/eval_candidate.sh`.
Legacy shape record: `checkpoints/v75_freeze.json`.

| Field | Legacy densest @ T=256 | Denser under gate @ T=128 |
|-------|------------------------|---------------------------|
| Shape | L23×d576 MQA h12/1 | e.g. L28×d576 or L36×d512 |
| Params | **~78.8M** (still max at T=256) | **~96–99M** |
| Serve | ternary Linear + int8 embed | same |
| Infer WS | ~30.22 / 30.40 MiB | must re-check `estimate_infer_ws` |

Train denser: `bash scripts/train_dense_t128_max.sh` (after GPU free).
Older lineage: `checkpoints/v2_dense` superseded; dual-ship pure_v2 is **not** sacred.

## Repository map

```
ll3m/           # Python package (train / model / QAT / tokenizer / eval)
  config.py        # GPTConfig, n_params(), choose_config
  model_torch.py   # v1/v2 GPT, estimate_infer_ws, estimate_budget
  train.py         # main train CLI (Lion, QAT, resume, early stop)
  qat.py           # STE int8/ternary + embed QAT + pack helpers
  tokenizer.py     # pure-Python domain BPE
  eval.py          # capability evals
c_infer/           # portable C serve (product claim)
  ll3m.c/.h     # model load, decode, L3 pin/warm
  main.c           # CLI --l3-resident
  Makefile         # -DCLM_QUANT=2 ternary default
scripts/
  download_data.sh
  prepare_data.py  # mix + BPE + .bin + mix_manifest.json
  export_weights.py
  train_dense_ternary.sh
data/              # sources, mix_*.{txt,bin}, tokenizer.json, manifest
checkpoints/       # runs; never commit huge blobs casually
docs/              # design + research (see docs/README.md)
tests/             # budget, product dense, checkpoint gates
```

## Tooling

- **Python env:** `uv` only (never pip / venv / ensurepip). Project venv: `.venv/`
- **Torch for this host:** ROCm build required for GPU train  
  (`torch==2.10.0+rocm7.0` from `https://download.pytorch.org/whl/rocm7.0`).  
  A CUDA-only wheel will report `cuda.is_available()==False` here.
- **GPU:** `export HIP_VISIBLE_DEVICES=0` (RX 7900 XTX)
- **Tests:** `uv run pytest` or `.venv/bin/pytest`
- **C build:** `make -C c_infer` (ternary Linears + int8 embeds product pack)

```bash
# Install / restore ROCm torch if needed
uv pip install --python .venv/bin/python 'torch==2.10.0+rocm7.0' \
  --index-url https://download.pytorch.org/whl/rocm7.0 \
  --index-strategy unsafe-best-match
```

## Data pipeline

Prefer **TinyStoriesV2-GPT4** (quality) over mixed GPT-3.5/4 slices.

```bash
bash scripts/download_data.sh          # default ~1.5 GiB train partial; MAX_BYTES=0 full
uv run python scripts/prepare_data.py --vocab-size 2048 --bpe-train-chars 40000000
```

`prepare_data.py` does: story split, incomplete-tail drop, quality filters, dedup,
train/val cross-dedup, 5% Shakespeare spice (train only), stratified BPE sample,
`mix_train.bin` / `mix_val.bin` / `mix_manifest.json`.

Current rebuild (see `data/mix_manifest.json`): ~**466M** train tokens, ~6M val tokens.

## Train recipes

### Phase 1 (fp32 capability) — product shape

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -u -m ll3m.train \
  --arch v2 --tokenizer-mode bpe \
  --n-layer 23 --n-embd 576 --n-head 12 --n-kv-head 1 --block-size 256 \
  --layer-scale-init 1e-4 \
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
  --batch-size 4 --grad-accum 16 --gpu-mem-fraction 0.45 \
  --dropout 0.1 --lr 3e-4 --warmup 2000 --steps 50000 --patience 0 \
  --device rocm --infer-scope ccd --serve-quant ternary \
  --qat none \
  --prompt "Once upon a time" \
  --out-dir checkpoints/v75
```

Or: `bash scripts/train_dense_ternary.sh` (phase 1 + phase 2).

### Phase 2 (serve quant)

```bash
# after phase 1 best exists
.venv/bin/python -m ll3m.train \
  --resume checkpoints/v75/best.pt \
  --qat ternary --embed-qat int8 \
  --lr 5e-5 --steps $((STEP+8000)) --patience 0 \
  --device rocm --serve-quant ternary \
  --out-dir checkpoints/v75_ternary
```

**Never start ternary QAT from random init** (STE noise). Resume fp32 best.

### Shape / L3 sizing

When changing architecture, always re-check:

```python
from ll3m.config import GPTConfig
from ll3m.model_torch import estimate_infer_ws
from ll3m.cache_info import probe_cache

cfg = GPTConfig(...)  # peri_ln, layer_scale_init, n_kv_head, etc.
gate = int(probe_cache().l3_bytes * 0.95)
inf = estimate_infer_ws(cfg, quant="ternary", embed_quant="int8")  # product default is ternary
assert inf.total <= gate
assert sum(p.numel() for p in GPT(cfg).parameters()) == cfg.n_params()
# GPTConfig library defaults (dropout 0, layer_scale 0) are NOT the freeze;
# set peri_ln / layer_scale_init / n_kv_head / intermediate from freeze JSON.
```

Objective for product search: **max `n_params` under gate**, with practical
heads `{8,12,16}` and `head_dim ∈ {32,40,48,64}` unless user asks for exotic max.

MQA (`n_kv_head=1`) usually wins param density vs GQA at T=256 under this gate.

## Hard lessons (do not regress)

1. **peri-LN on L23×d576 blocked early learning** (val stuck ~5.9). Product freeze
   keeps peri **off**. Smaller L18×d320 could use peri; do not re-enable blindly.
2. **LayerScale `1e-2` is too aggressive here**; product uses **`1e-4`**.
3. **Speed probe must not call `opt.step()`** (corrupts init + Lion moments).
   Fixed in `train.py`; keep it that way.
4. **Embed ternary after QAT is worse quality** than int8 embeds for tiny L3 win
   (~0.5 MiB). Product: **Linear ternary + embed int8**.
5. **Early stop (`patience>0`) can kill a long run** before capability emerges.
   Long product runs use **`patience 0`**.
6. Default train log loss includes **MTP** terms; val CE from `eval_loss` is pure CE.
7. Disk is tight on this host; avoid dumping multi-GiB scratch under `/tmp` (tmpfs).

## Coding conventions

- Prefer **minimal, readable** changes. No drive-by refactors.
- **Typed** Python; illegal states unrepresentable where cheap.
- Prefer **stdlib / in-process APIs** over shelling out (`curl` only for downloads).
- Python packages: **`uv`**, never `pip`.
- No em dashes in new prose (chat, commits, docs, comments).
- No AI attribution in commits, PRs, comments, or docs.
- Match existing style in the file you edit.
- Update `docs/` when product freeze, quant pack, or train recipe changes.

## C inference (default generate/chat path)

Python chat/generate is a thin wrapper: BPE + UI in Python, **decode in `ll3m_c`**.
Torch in-process generate is opt-in only: pass `--python` (fallback).

```bash
make -C c_infer -j
# export .pt → L3T2 ternary pack (product default; ~20 MiB weights)
uv run python scripts/export_weights.py \
  --checkpoint checkpoints/v75/best.pt --out checkpoints/v75/best.ll3m --quant ternary
# debug f32: --quant f32  (L3M2, does not fit L3)

# preferred: Python UI wraps C
uv run python -m ll3m.chat --checkpoint checkpoints/v75/best.ll3m complete \
  --prompt "Once upon a time" --tokens 80
# Q&A + tools (default chat mode)
uv run python -m ll3m.chat --checkpoint checkpoints/v75/best.ll3m chat --mode agent
# story continuation
uv run python -m ll3m.chat --checkpoint checkpoints/v75/best.ll3m chat --mode story
# tool SFT (after v75, optional stage-1)
# bash scripts/train_tool_sft.sh

# goldens + residency + capability
uv run python scripts/compare_c_golden.py --checkpoint checkpoints/v75/best.pt --quant ternary
uv run python scripts/idea_bench_c.py --weights checkpoints/v75/best.ll3m --tokens 32
uv run python scripts/capability_report.py --checkpoint checkpoints/v75/best.pt

# direct C CLI (L3T2; BPE ids from Python)
./c_infer/ll3m_c -m checkpoints/v75/best.ll3m --token-ids 12,34,56 -n 120 \
  --residency-hard --warm 16 --ccd 0
```

| Flag | Effect |
|------|--------|
| `--l3-resident` | CCD/isol bind, mlock arenas, warm, refuse if WS > CCD |
| `--residency-hard` | + mlockall, **1 core** (isolcpus preferred), SCHED_FIFO, MAP_HUGETLB 2MiB if reserved, hierarchical touch |
| `--steady-warm N` | Decode-only steps **before** PMU (default 64 hard / 16 soft) so DRAM% is steady-state |

Host: `scripts/residency_host_setup.sh status|hugepages|cmdline-hint`  
Access: L1 dense pad; L2 tiled GEMV + software weight prefetch
(`CLM_W_PREFETCH_AHEAD=32` NTA default; see `docs/READAHEAD_DESIGN.md`); L3 sequential 64B.
Multi: `c_infer/multi_bench` pthread product; optional `--burst K` RR. Product WS gate **30.4 MiB** CCD.

## Tests worth running

```bash
uv run pytest tests/test_budget.py tests/test_product_dense.py -q
# after architectural change, also:
uv run pytest tests/ -q
```

`test_product_dense.py` encodes the **v75** freeze (L23×d576 MQA, 78,815,712 params,
peri off, ls 1e-4, ternary+int8 WS under CCD gate). Update it when the product shape
freeze changes. Older L18×d320 peri-on expectations are retired.

## Status / ops

| Artifact | Use |
|----------|-----|
| `checkpoints/v75/progress.json` | live step, loss, best_val, tok/s |
| `checkpoints/v75/train.log` | full log + evals |
| `checkpoints/v75/best.pt` | best val CE weights |
| `pgrep -af 'll3m.train'` | is train alive? |
| `rocm-smi` | GPU util |

## Quant / BitNet-inspired backlog

Do **not** copy full BitNet from-scratch pretrain or pure 1-bit (no zero). Keep
ternary Linears + int8 embeds as the product pack.

Full survey + paper IDs: **`docs/RESEARCH_MODELS_2026.md`**.

| # | Item | Notes |
|---|------|--------|
| 1 | **v75 finish → ternary QAT + int8 embed QAT** | `scripts/run_v75_ternary_qat_when_ready.sh` (waiter) → `checkpoints/v75_ternary/` |
| 1b | **Native ternary from step 0** | Local ROCm → `checkpoints/v75_native_ternary/`. spark2 CUDA → pull into **`checkpoints/v75_native_ternary_spark2/`** only (never overrides local). Scripts: `train_v75_native_ternary*.sh`, `pull_spark2_native_when_ready.sh`. |
| 1c | **Tool SFT after native** (keep QAT on) | `CKPT=.../best.pt bash scripts/train_tool_sft.sh`; When2Call mix via `gen_tool_sft.py` |
| 1d | **Engram-lite host KB** | `ll3m/kb_facts.json` **v2** (entries+aliases+flat); `tools.lookup` word-bound |
| 1e | **Dense knowledge ladder** (active) | spark1 `ll3m-dense`: A KB → B atomic mix+MinHash → C continue QAT → D fact+tool SFT QAT → MC vs Qwen4. Doc: **`docs/DENSE_KNOWLEDGE_LADDER.md`**. Launch: `bash scripts/launch_spark1_dense.sh` |
| 2 | **Serve-grid quality compare** | post-hoc fp32 export vs QAT: `scripts/compare_fp32_vs_qat.py` → `quality_compare.json`; also native vs fp32/posthoc |
| 3 | **Train–serve grid lock** | same absmean / pack / embed int8 in train STE and C `L3T2` (lossless BitNet-style; bitnet.cpp I2_S lesson) |
| 4 | **`BitLinear` module** | first-class STE absmean Linear (replace monkey-patch QAT) |
| 5 | **Longer QAT polish** | only if (2) shows a large gap; lower LR, more steps |
| 6 | **Optional W1.58A8** | int8 activation STE + C match; **after** weight QAT is good |
| 7 | **Optional TL/bitnet GEMV** | ternary LUT kernel if still CPU-bound after (1–3); see T-SAR / bitnet.cpp |

Skip for now: pure binary weights, from-scratch ternary-only pretrain, trit packing.

## Research follow-ups (from `docs/RESEARCH_MODELS_2026.md` §5)

Do **after** quant backlog 1–2 establish a QAT baseline. Do **not** change the
L23×d576 MQA freeze for these.

| ID | Item | Inspiration | Success metric |
|----|------|-------------|----------------|
| R-B | Gold tool-SFT mix (calc/lookup/time + no-call ~1:1) | Kimi / BFCL / When2Call | Agent mode works with `--no-assist-tools` |
| R-C | KD from Qwen3-0.8B/1.7B or Ministral-3B into v75 shape | Ministral cascade, Nemotron Elastic | Val CE + samples vs CE-only |
| R-D | Data curriculum: story → short Q&A → tools | SmolLM3, Phi | Capability report + tool accuracy |
| R-E | Constrained decode for `<call>…</call>` | Colibrì grammar-draft + SLM tool papers | **Partial:** `ll3m/constrained.py` clip/repair + agent; optional C-side mask later |
| R-H | Host Engram-lite: hashed fact table for `lookup` | DeepSeek Engram arXiv:2601.07372 | Hit rate; no weight growth |
| R-I | Parallel host tool fan-out on multi-call | Kimi Agent Swarm (host-side) | Wall latency multi-tool |
| R-J | Softcap / QK clamp ablation during QAT | Gemma softcap, Kimi QK-Clip | Loss spikes, final CE |

**Skip (documented anti-patterns):** MoE at 79M, MLA/DSA/Mamba at T=256, DiffusionGemma C path, packing world knowledge into ternary weights (see `docs/KNOWLEDGE_DENSITY_SMALL_MODELS.md`).

Long GPU trains: launch with `setsid` + log redirect; do not block the agent
session on full 50k-step wall time.

**ROCm torch footgun:** `uv sync` / PyPI can replace the AMD wheel with a CUDA
build (`torch==…+cu*`) so `torch.cuda.is_available()` becomes False on the 7900
XTX. Phase-1 used `torch==2.10.0+rocm7.0`. Restore with:

```bash
uv pip install --python .venv/bin/python 'torch==2.10.0+rocm7.0' \
  --index-url https://download.pytorch.org/whl/rocm7.0
```

`scripts/run_v75_ternary_qat_when_ready.sh` now aborts with this fix command if
GPU torch is missing.

## Doc index

| Doc | When to read |
|-----|----------------|
| `docs/OVERVIEW.md` | product framing |
| `docs/ARCHITECTURE_V2.md` | v2 block details |
| `docs/TARGET_75M.md` | shape search + freeze |
| `docs/PRODUCT_CANDIDATE.md` | current ship candidate |
| **`docs/SHIP_FREEZE.md`** | **settled dual SKU + residency/setcap measures** |
| `docs/QUANTIZATION.md` | ternary / int8 / QAT |
| `docs/L3_RESIDENCY.md` | residency proof protocol |
| **`docs/READAHEAD_DESIGN.md`** | **L2 tile / multi-KV readahead; R1 knobs + R4 `--burst` measured** |
| **`docs/AGENT_BATTERY_SUITE.md`** | **101-case agent suite, gate_stats, pure dual scores** |
| **`docs/RESEARCH_MODELS_2026.md`** | **deep model + arXiv survey → actions (canonical)** |
| **`docs/HF_SURVEY_LEARN_STEAL.md`** | **HF popular/trending (2026-07-24) papers/blogs → ranked steal/skip** |
| **`docs/ARXIV_SURVEY_LEARN_STEAL.md`** | **arXiv pass (2026-07-24) ternary/agent/memory/distill → steal/skip + IDs** |
| `docs/KNOWLEDGE_DENSITY_SMALL_MODELS.md` | small models + knowledge compression → LL3M |
| **`docs/DENSE_KNOWLEDGE_LADDER.md`** | **active denser rebuild A–D (KB v2, MinHash, QAT lock, spark1, vs Qwen4)** |
| **`docs/DENSE_CAPABILITY_PLAN.md`** | **max density stack P0–P4 (product gate, Engram, long dense, DPO)** |
| **`docs/ROADMAP.md`** | **phased roadmap P0–P5 (measure → ship → GRPO → KD → quant → paper)** |
| `docs/BASE_ENGLISH.md` | product language goal (base English over stories) |
| `docs/RESEARCH.md` | earlier architecture tweak map |
| `docs/TOOL_CALLING.md` | tool protocol + agent SFT + KB v2 |
| `docs/EXPERIMENTS.md` | active experiment log (incl. dense ladder) |

## What not to do

- Do not claim train WS fits L3.
- Do not grow vocab to 32k+ (embeds steal the CCD).
- Do not add MoE / MLA / long-context tricks without a residency plan.
- Do not force-push, amend published history, or delete checkpoint trees without
  explicit user ask.
- Do not install packages system-wide or with `--break-system-packages`.
- Do not treat frontier MoE/1M-ctx papers as a shopping list for the L3 product
  (see RESEARCH_MODELS_2026 decision table).
