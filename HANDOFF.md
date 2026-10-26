# Handoff — resume here

**Written:** 2026-08-10 ~18:36 +08 (updated after v78 SciQ train complete)  
**Repo:** `/home/maci/Desktop/cachelm`  
**Host:** Ryzen 9 9950X, RX 7900 XTX (ROCm), L3 CCD 32 MiB, product gate **30.4 MiB**

Read this first in a new session. Refresh live state with §1. Canonical fairness note: **`docs/BINARY_VS_TERNARY_LADDER.md`**.

---

## 0. One-paragraph mission

Product ship remains **ternary L3T2** (~79M, L23×d576, agent + SciQ dual-ship). Research path: **binary** denser pack → more params under L3 (**v78 ~195M**), thin ladder base → multi-mix → SciQ denser → eval. **Do not** treat binary SciQ ~22.5% vs ternary p1 **69.5%** as pack-alphabet failure: ternary had far more stages (see fairness doc). Side work: **semantic compressor** scaffold + literature. Next: finish agent battery, write DECISION, decide binary follow-up or stop thrash.

---

## 1. Live status (refresh)

```bash
pgrep -af 'll3m.train|run_binary_c195|eval_agent_battery'
cat checkpoints/v78_binary_c195_sciq/progress.json
cat checkpoints/v78_binary_c195_sciq/std_benchmarks_sciq400.json
ls -la checkpoints/v78_binary_c195_sciq/
tail -50 checkpoints/v78_binary_c195/ladder.log
rocm-smi | head -15
```

### Snapshot at this handoff write

| Stage | Path | Result | State |
|-------|------|--------|--------|
| Base | `checkpoints/v78_binary_c195/` | 60k done, best val **1.145** | **done** |
| Multi-mix SFT | `checkpoints/v78_binary_c195_sft/` | 72k done, best val **0.600** | **done** |
| SciQ denser | `checkpoints/v78_binary_c195_sciq/` | **86k/86k** done | **done** |
| SciQ@400 | `…/std_benchmarks_sciq400.json` | **22.5%** (90/400) | **done** |
| Export L3B2 | `…/best.ll3m` | ~26 MiB file | **done** |
| C smoke | `…/c_smoke.log` | WS **29.56 MiB**, fill 92.4%, **fits CCD** | **done** |
| Agent battery | `eval_agent_battery.py` PID ~1797812 | writing `battery_product_gate_on.json` | **RUNNING** |
| Ladder wrapper | `run_binary_c195_ladder.sh` | still open until battery+DECISION | **running** |

| Metric | Value |
|--------|--------|
| Shape | L34×d576 h12/1 mid2880 T=256 V=2048 |
| Params | **194,908,416** |
| Quant | binary Linear + int8 embed (L3B2) |
| Train flag | `--force` (estimator overstates act) |
| C weights / total | **25.52 / 29.56 MiB** (headroom ~2.4 MiB vs 32 CCD) |
| SciQ@400 seed 42 | **22.5%** (random 25%) |
| v77 sciq | **19.5%** |
| Ternary p1 SciQ | **69.5%** (not fair same recipe) |

**Train PID for SciQ is gone** (finished). Battery is the only heavy job.

---

## 2. Immediate next actions

1. **Wait for battery** → `checkpoints/v78_binary_c195_sciq/battery_product_gate_on.json`  
2. When ladder exits, confirm `DECISION.txt` (or write it; template §8).  
3. Update this handoff “snapshot” line if battery score is in.  
4. **Do not promote** binary over ternary ship.  
5. Optional follow-ups only if user asks: longer SciQ densify, semcomp denser mix, more agent multi-mix, or archive binary research.

### Suggested decision (pre-battery)

| Evidence | Implication |
|----------|-------------|
| SciQ 22.5% after full thin ladder | +params helped a little vs v77 (19.5); still near-random |
| C pack fits 30.4 gate | Capacity claim **OK** |
| Fairness | Need ternary-depth recipe or abandon SciQ race on binary |
| Product | Ship stays **ternary** agent + SciQ twins |

---

## 3. v78 recipe (reproduce)

| | |
|--|--|
| Freeze JSON | `checkpoints/v78_binary_c195_freeze.json` |
| Base script | `scripts/train_dense_binary_c195.sh` |
| Ladder | `scripts/run_binary_c195_ladder.sh` |
| Base data | `data/mix_train.bin` 60k steps, lr 3e-4, bs 4 accum 16 |
| SFT | `data/cap_mix_v15_agent.txt` +12k, lr 5e-6 |
| SciQ | `data/cap_mix_v16_binary_sciq.txt` +15k (to step 86k), lr 5e-6 |
| Always | `--qat binary --embed-qat int8 --serve-quant binary --force` |

```bash
# Manual eval if ladder dies mid-battery
CKPT=checkpoints/v78_binary_c195_sciq/best.pt   # or last.pt
OUT=checkpoints/v78_binary_c195_sciq
.venv/bin/python scripts/eval_agent_battery.py \
  --checkpoint "$OUT/best.ll3m" --no-assist --temperature 0 \
  --out "$OUT/battery_product_gate_on.json"
```

---

## 4. Fairness: ternary got much more work

Full write-up: **`docs/BINARY_VS_TERNARY_LADDER.md`**.

Short version:

| | Ternary ship/twins | Binary v77/v78 |
|--|--------------------|----------------|
| Pretrain | Long fp32 + QAT phases | Native binary STE base only |
| Dense ladder | Atomic, MinHash, continue QAT, multi stages | No |
| Dual ship | Agent multi-mix → 101/101 + SciQ p1 69.5 | Thin multi-mix + short SciQ |
| Tool/agent polish | Many caps, gate rewrites, Engram | One multi-mix stage |
| SciQ@400 | **~69.5** (p1) | **~19–22.5** |
| Fair pack test? | — | **Not yet** |

Comparing 22.5 vs 69.5 is **recipe depth**, not proof that binary alphabet cannot learn SciQ.

---

## 5. Product freeze (unchanged)

| | |
|--|--|
| Shape | L23×d576 MQA mid1536, peri **off**, ls **1e-4** |
| Params | ~78.8M |
| Serve | Linear **ternary** + embed **int8** L3T2 |
| Gate | ~30.22 MiB est / 30.4 product |
| Agent ref | cap7 / pure_v2 multi-mix lineage |
| SciQ twin | `checkpoints/v76_dense_c_p1` ~69.5 |

See `AGENTS.md`, `docs/PRODUCT_CANDIDATE.md`.

---

## 6. Semantic compressor (scaffolded)

| | |
|--|--|
| Doc | `docs/SEMANTIC_COMPRESSOR.md` (§1b knowledge theory, §3 related work, §14 refs) |
| Code | `ll3m/semcomp/` |
| Tests | `uv run pytest tests/test_semcomp.py -q` |
| Tiny smoke | `scripts/semcomp_tiny_slice.py` → `data/semcomp_tiny/` (~0.44× BPE, claim gate ok) |
| Not done | Train A/B; P2 LLM extract; feed into v78/v79 recipe |

---

## 7. L3 capacity / binary densify

| | |
|--|--|
| Doc | `docs/L3_CAPACITY_SEARCH.md` |
| Insight | Estimator overstates acts; C is authority |
| v77 142M | C total ~23 MiB |
| v78 195M | C total **29.56 MiB** (measured sciq export) |
| Scripts | `scripts/search_l3_capacity.py`, train scripts above |

---

## 8. DECISION.txt template (write when battery done)

```text
v78_binary_c195 ladder
shape=L34×d576 mid2880 ~194.9M binary+int8
base_val=1.145  sft_val=0.600  sciq_tracker_best=0.600 (domain mismatch)
SciQ@400=22.5  (v77_sciq=19.5  ternary_p1=69.5 unfair)
C_ws=29.56 MiB weights=25.52 MiB fits_CCD=YES
battery=TODO
promote=NO  ship stays ternary
note=thin ladder vs ternary dual-ship depth; capacity OK quality not
```

Path: `checkpoints/v78_binary_c195_sciq/DECISION.txt`

---

## 9. Key paths

```
HANDOFF.md
docs/BINARY_VS_TERNARY_LADDER.md
docs/SEMANTIC_COMPRESSOR.md
docs/L3_CAPACITY_SEARCH.md
AGENTS.md
scripts/run_binary_c195_ladder.sh
scripts/train_dense_binary_c195.sh
ll3m/semcomp/
checkpoints/v78_binary_c195/
checkpoints/v78_binary_c195_sft/
checkpoints/v78_binary_c195_sciq/     ← hot eval
checkpoints/v77_binary_l30*/          ← prior binary
checkpoints/v76_dense_c_p1/           ← SciQ twin ternary
checkpoints/v76_dense_c_pure_v2/      ← agent ship family
data/semcomp_tiny/
```

---

## 10. Paste-ready new session prompt

```text
Resume from HANDOFF.md and docs/BINARY_VS_TERNARY_LADDER.md in /home/maci/Desktop/cachelm.

1) Check if agent battery finished; if yes, read battery JSON + write DECISION.txt.
2) Do not promote binary over ternary ship (SciQ 22.5 vs p1 69.5 is unfair recipe depth).
3) Capacity claim for v78 L3B2 is OK (C WS ~29.6 MiB).
4) Semcomp exists; no train A/B unless asked.
5) Update HANDOFF.md snapshot when battery completes.
```

---

## 11. Arc changelog

| When | What |
|------|------|
| Prior | Ternary dual-ship, Engram, dense ladder, many caps |
| v77 | Binary 142M thin ladder; SciQ ~17–19.5% |
| 2026-08-07 | C headroom math; v78 ~195M train start |
| 2026-08-10 AM | Base + SFT done; SciQ denser running |
| 2026-08-10 | semcomp scaffold + research/refs; HANDOFF v1 |
| 2026-08-10 ~17:28 | SciQ train done; SciQ@400 **22.5%**; L3B2 C fit |
| 2026-08-10 ~18:36 | Battery still running; handoff rewritten + fairness doc |

**Update this file** when battery completes or priorities change.
