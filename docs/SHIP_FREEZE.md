# Ship freeze (settled dual SKU)

**Date:** 2026-07-29 (host policy refresh **2026-08-01**)  
**Status:** product ship frozen; short CE/DPO thrash **stopped**.  
**Host policy:** **no boot `isolcpus`** on this machine (desktop path only).  
**Live status JSON:** `checkpoints/SHIP_STATUS.json`

## Product thesis

Inference working set (weights + acts + KV) fits **one CPU L3 CCD** (~30.4 MiB gate on 32 MiB).  
Train may use full DRAM/GPU. Serve claim is portable **C** (`L3T2` ternary Linears + int8 embeds).

| Gate | Value |
|------|------:|
| L3 per CCD | 32 MiB (Ryzen 9 9950X) |
| Product budget | 95% ≈ **30.4 MiB** |
| Design estimator | ~**30.22 MiB** (`estimate_infer_ws`) |
| C runtime WS (measured) | ~**24.32 MiB** |
| On-disk pack | ~**20.3 MiB** `.ll3m` |

Shape freeze: **L23 × d576**, heads **12/1 MQA**, T=256, V=2048, peri_ln off, LayerScale 1e-4.

## Dual ship (weights)

| Role | Pack | Metric |
|------|------|--------|
| **Product agent** | `checkpoints/v76_dense_c_pure_v2/best.ll3m` | gate-on **101/101** (2026-07-29 battery) |
| **SciQ peak** | `checkpoints/v76_dense_c_p1/best.ll3m` | SciQ **69.5**, macro ~41.4 |
| Pure research (optional) | `checkpoints/v76_dense_c_p3b/best.ll3m` | pure ~**93/101** |

CLI defaults: **assist off**, **gate on**. Pure stress: `--pure-policy`.

### Product agent policy

| Flag | Meaning |
|------|---------|
| `assist_tools=False` | Model must emit `<call>`; host does not invent silent calls |
| `gate_tools=True` | Block tools on chitchat; **rewrite wrong tool name** when host intent is known (e.g. model says `lookup` for “what day is it” → `time`) |
| Pure (`gate=False`) | Model-only stress; product scores are **model × gate** |

Not claimed: one pure-policy pack maxing tools + SciQ + pure no_call under short QAT.

## Host Engram-lite

| Field | Value |
|-------|------:|
| Path | `ll3m/kb_facts.json` (v2 entries + flat) |
| Entries / keys | **1022** / **3057** (wave2 2026-07-30) |
| Expand | `uv run python scripts/expand_engram_kb.py` |
| Lookup | word-bounded match (no `pi` ⊂ `pizza`) |

Host facts are free density vs ternary weight bits. Grow Engram before packing rare facts into weights.

## C serve + residency

### Preferred measure path (this host)

```bash
# one-time (survives rebuild if binary inode not replaced; re-run after make)
sudo setcap cap_sys_nice+ep c_infer/ll3m_c

# full report (BPE + hard residency + PMU)
uv run python scripts/measure_residency.py --ccd 1 --cpu 15 -n 256

# or wrapper
LL3M_CPU=15 LL3M_CCD=1 scripts/run_residency_hard.sh \
  -m checkpoints/v76_dense_c_pure_v2/best.ll3m \
  --token-ids … -n 256 --greedy
```

Host helpers:

| Command | Role |
|---------|------|
| `scripts/residency_host_setup.sh status` | isol / hugepages / memlock |
| `… topology` | dual CCD map |
| `… protocol` | path to GOOD DRAM% |
| `… check-caps` | FIFO / setcap |
| `… cmdline-hint` | other machines only (product host: **no isolcpus**) |
| `… quiet-cpu` | prints **15** |

### Measured (2026-07-28…29, product pack)

| Item | Result |
|------|--------|
| Size-fit | **PASS** 24.32 MiB / 32 MiB CCD |
| mlock | **100%** model + state |
| `cap_sys_nice` | **ep** after setcap |
| SCHED_FIFO | **OK** prio 25 |
| tok/s hard+FIFO | **~280–282** (n=256) |
| boot isolcpus | **out of scope** (host policy: no isolcpus) |
| DRAM% CCD0 (old first-CPU pin) | ~**35%** WEAK |
| DRAM% CCD1 quiet core | ~**9–20%** WEAK (varies; accepted on desktop) |
| Product DRAM claim | **not** GOOD ≤5%; size-fit + mlock + FIFO tok/s only |

Engine pin policy (no isolcpus): `--cpu` / `CLM_PIN_CPU` → single-CPU taskset affinity → **quiet physical core of CCD** (not CPU0). Prefer `--ccd 1` on dual-CCD 9950X. Flags: `--ccd N --cpu N`.

### Labels (honest)

| Label | Condition |
|-------|-----------|
| Size-fit PASS | runtime WS &lt; one CCD |
| mlock PASS | mincore 100% |
| DRAM WEAK | size OK; L2-fill DRAM share high (normal without isolcpus) |
| DRAM GOOD | ≤5% after steady warm (not a ship gate on this host) |

## Paper

| Artifact | Path |
|----------|------|
| Markdown | `docs/paper/LL3M_L3_Resident_Ternary_SLM.md` **v0.7.9** |
| TeX / PDF | `docs/paper/ll3m_l3_resident.{tex,pdf}` **v0.7.9** |
| Build | `cd docs/paper && tectonic -X compile ll3m_l3_resident.tex` |
| Agent suite | `docs/AGENT_BATTERY_SUITE.md` (gate_stats, pure dual scores) |
| Readahead | `docs/READAHEAD_DESIGN.md` (prefetch knobs; multi `--burst`) |

## What not to do

- Short CE joint thrash or naive long DPO (tools/fluency collapse).
- Claim train WS fits L3.
- Claim pure-policy 99/101 or GOOD (≤5%) DRAM% on this host (no isolcpus).
- Grow vocab to 32k+ (embeds steal the CCD).
- Force-push / delete checkpoint trees without explicit ask.

## Capability plan reference

`docs/DENSE_CAPABILITY_PLAN.md` — P0–P3 settled dual ship.  
Optional next (non-thrash): more Engram, KD/teacher (R-C), soft IRQ quiet if DRAM% must improve without reboot. **No isolcpus work.**
