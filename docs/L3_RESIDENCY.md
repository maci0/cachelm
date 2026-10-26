# L3-only inference residency

## Claim

During **timed C inference**, after pin and warm-up:

1. **Capacity:** `weights + activations + KV` fit in **one L3 CCD** (32 MiB on the 9950X).  
2. **Residency:** that set does **not** thrash to DRAM. L2 misses are filled almost
   entirely from **local L3** (measured with AMD `l2_fill_rsp_src.*`).

The binary + libc cold paths may touch DRAM at process start. The claim is about
the **steady-state** prefill/decode path **after** warm-up.

## Working set contents (cache tiers)

| Tier | Size (this host) | Contents |
|------|------------------|----------|
| **L1i** | 32 KiB | Hot kernels (`.text.hot`: GEMV, decode, RMSNorm) |
| **L1d** | 48 KiB | Decode pad: one-token `x/xb/qkv/mlp` (mlock'd, separate arena) |
| **L2** | 1 MiB | Decode attention scores + software-prefetched weight tiles |
| **L3** | 32 MiB CCD | Full **weights** + **KV** + prefill scratch + logits |

| Arena | Contents |
|-------|----------|
| Model (L3) | Embeds, norms, **quantized** Linears, scales |
| State L3 | Prefill buffers, logits, **K/V cache**, tokens |
| State L1d | Single-token residual + work vectors |
| State L2 | Per-head decode scores |

GEMV keeps the activation vector L1-hot and streams weight rows with software
prefetch (`CLM_W_PREFETCH_HINT=0` → `_MM_HINT_NTA` default; optional T1/T0)
so tiles land in L2 while the bulk of weights stay L3-resident.

## Mechanisms

| Step | Implementation |
|------|----------------|
| Contiguous arenas | `posix_memalign` + zero |
| **mlock** | Default on (disable: `-DCLM_NO_MLOCK`) |
| Raise memlock rlimit | Best-effort `RLIMIT_MEMLOCK` |
| Prefault | Touch every 64-byte line |
| THP hint | `MADV_HUGEPAGE` (best-effort) |
| One CCD | `clm_bind_l3_ccd(n)` from sysfs `shared_cpu_list` |
| Warm | `clm_warm(N)` full forwards |
| Oversize refuse | `--l3-resident` exits 2 if WS > CCD |

## How to run

```bash
cd c_infer
make
ulimit -l unlimited   # if mlock fails

./ll3m_c -m ../checkpoints/best_L6.clm \
  -p $'ROMEO:\n' -n 200 --greedy \
  --l3-resident --warm 12 --ccd 0

# Hard residency (recommended for measurement):
# mlockall + single-core CCD pin (private L2) + SCHED_FIFO + hugepage madvise
# + hierarchical L1→L2→L3 touch patterns before timed decode
./ll3m_c -m ../checkpoints/best_L6.clm \
  -p $'ROMEO:\n' -n 200 --greedy \
  --residency-hard --warm 16 --ccd 0
# --no-cache-stats  to skip PMU section

# Extra hardware proof (external perf wrap)
./measure_cache.sh ../checkpoints/best_L6.clm 200
DRAM_MAX_PCT=2 ./measure_cache.sh ../checkpoints/best_L6.clm 200   # stricter
```

### Built-in stats (no external `perf` needed)

| Section | Meaning |
|---------|---------|
| L3 occupancy | Planned WS / one CCD L3 (how much of the cache the model **claims**) |
| RAM residency | `mincore` page presence after mlock (eligible for L3) |
| Cache PMU | **Steady decode only** (after `--steady-warm`); hit/miss + L2 fills |

### Host setup (items 1–2)

```bash
# status: isolcpus? hugepages? caps?
bash scripts/residency_host_setup.sh status
bash scripts/residency_host_setup.sh topology      # L3 CCD shared_cpu_list
bash scripts/residency_host_setup.sh protocol      # path to GOOD DRAM%
bash scripts/residency_host_setup.sh check-caps    # SCHED_FIFO / cap_sys_nice
# cmdline-hint exists for other machines only; product host: no isolcpus

# reserve 2MiB pages for MAP_HUGETLB model arena (root)
sudo bash scripts/residency_host_setup.sh hugepages 64   # 128 MiB

# without reboot (weaker): soft IRQ steer + FIFO cap
sudo scripts/residency_soft_isolate.sh "$(bash scripts/residency_host_setup.sh quiet-cpu)"
sudo setcap cap_sys_nice+ep c_infer/ll3m_c
```

**9950X-shaped dual CCD:** CCD0 CPUs `0-7,16-23`; CCD1 `8-15,24-31`.  
`cmdline-hint` isolates one physical core + SMT sibling (default quiet CPU **15** + **31**).

Engine side: `--residency-hard` prefers isolated CPUs if present, else 1 core of a CCD;
tries `mmap(MAP_HUGETLB|MAP_HUGE_2MB)` for model/state, falls back to 4K+THP.

### Path to GOOD DRAM% (≤5% L2 fills from DRAM)

| Layer | Role |
|-------|------|
| Size | WS ≤ one CCD (design 30.4 / runtime ~24.3 MiB) |
| Pin | mlock / mlockall (`--residency-hard`) |
| Quiet core | soft IRQ steer + `taskset` / `--cpu` (no isolcpus on this host) |
| RT | `SCHED_FIFO` via `cap_sys_nice` or root |
| Steady PMU | `--steady-warm` + long `-n` (≥128) |
| Optional | hugepages ≥32 × 2 MiB |

```bash
# preferred product measure (hard+FIFO, product agent pack)
LL3M_CPU=$(bash scripts/residency_host_setup.sh quiet-cpu) \
  scripts/run_residency_hard.sh \
  -m checkpoints/v76_dense_c_pure_v2/best.ll3m -n 256 --greedy
```

| Label | Meaning |
|-------|---------|
| Size-fit PASS | runtime WS < 32 MiB CCD |
| mlock PASS | mincore 100% model+state |
| DRAM GOOD | L2-fill DRAM share ≤5% after steady warm |
| DRAM WEAK | size OK, DRAM share high (noise / no FIFO) |
Full checklist text: `scripts/residency_host_setup.sh protocol` (cmdline-hint kept for other machines only).

### Access patterns (item 3)

| Tier | Pattern |
|------|---------|
| L1d | Dense R+W on one-token pad; GEMV keeps `x` hot |
| L2 | Output-tiled ternary GEMV + software weight prefetch (see knobs below) |
| L3 | Sequential 64B streams of weights + KV |

### Software prefetch knobs (product C, `ll3m_v2.c`)

| Macro | Default | Meaning |
|-------|--------:|---------|
| `CLM_W_PREFETCH_AHEAD` | 32 | packed-byte offset in ternary row loops |
| `CLM_W_ROW_PREFETCH` | 64 | bytes into current/next rows |
| `CLM_W_PREFETCH_HINT` | 0 | 0=NTA, 1=T1 (L2), 2=T0 (L1) |
| `CLM_NEXT_LAYER_PREFETCH` | 1 | kick next layer Wq/Wk/Wgate/Wup at layer end |

Grid: `bash scripts/prefetch_grid.sh` (2026-08-03: **no keep**; defaults retained).  
Design + multi RR: [READAHEAD_DESIGN.md](READAHEAD_DESIGN.md).

### Steady PMU window (item 5)

```
full warm (forward×N) → steady-warm (decode×M, not measured) → PMU on → timed -n tokens
```

Defaults with `--residency-hard`: `--warm 24 --steady-warm 64`.

## How to read `perf`

AMD L2 fill sources (where L2 misses are satisfied):

| Event | Meaning |
|-------|---------|
| `local_ccx` | **This CCD’s L3** (desired) |
| `near_cache` / `far_cache` | Other on-chip caches |
| `dram_io_near` / `dram_io_far` | **DRAM** (want ~0% after warm) |

`measure_cache.sh` prints **DRAM share of fills** and:

- `L3_RESIDENCY_GATE: PASS` if WS fits CCD **and** DRAM share ≤ `DRAM_MAX_PCT` (default 5)  
- `FAIL` otherwise  

**L1 miss rate can be high** (streaming GEMV). That is L1↔L2/L3 traffic, not DRAM,
if fill sources stay on-chip.

### Example (L6 ternary on this host)

```
infer WS:     3.35 MiB / 32 MiB CCD
mlock:        locked
CCD bind:     OK
L2 fills:     ~97% local L3, ~3% DRAM
L3_RESIDENCY_GATE: PASS
```

## What absolute “zero DRAM” cannot mean

Counters may still tick for:

- process startup before warm  
- `perf` attach  
- rare OS noise  

Engineering claim: **after pin+warm, steady decode data traffic is L3-resident;
DRAM share is noise-level.**

## Density helps

Ternary Linears (`-DCLM_QUANT=2`, default) shrink weight storage vs int8, leaving
L3 headroom for acts/KV or larger models.

| L6d192 | Weights | Infer WS |
|--------|---------|----------|
| int8 | ~2.8 MiB | ~5.3 MiB |
| ternary | ~0.9 MiB | ~3.4 MiB |

## New-model checklist

1. C print / `estimate_infer_ws`: WS ≤ 32 MiB CCD  
2. `./ll3m_c ... --l3-resident` succeeds  
3. `make cachestat` → `L3_RESIDENCY_GATE: PASS`  
4. Record WS MiB + DRAM% in release notes  

## Applied on this host (ops)

| Item | Status |
|------|--------|
| `vm.nr_hugepages=64` (2MiB) | set + `/etc/sysctl.d/99-ll3m-hugepages.conf` |
| memlock unlimited | `/etc/security/limits.d/99-ll3m-residency.conf` (new logins) |
| Soft IRQ steer off CPU 31 | `scripts/residency_soft_isolate.sh 31` |
| irqbalance | stopped if it was running (restart if you want it back) |

Quick measure (product pack):

```bash
# FIFO without full root (once):
sudo setcap cap_sys_nice+ep c_infer/ll3m_c

scripts/run_residency_hard.sh \
  -m checkpoints/v76_dense_c_pure_v2/best.ll3m -n 256 --greedy
```

Product host: **no isolcpus**. Prefer hard+FIFO + quiet `--cpu` / CCD1; DRAM% may stay WEAK.

## Single-stream throughput (product denser-C L3T2, 2026-07-27)

| Mode | tok/s | DRAM L2-fill share | Notes |
|------|------:|-------------------:|-------|
| Soft `--l3-resident` (CCD multi-core) | ~260–270 | often ~30–40% WEAK | not faster than hard here |
| Hard + `SCHED_FIFO` (`cap_sys_nice`) | **~280–282** | **GOOD ≤5%** on long windows *when core is quiet* | preferred single-stream |
| Early unisolated logs (paper v0.6) | ~294 soft / ~269 hard | ~24–27% WEAK | pre-FIFO / pre-bind fixes |
| Hard+FIFO CCD0 first core (pre-2026-07-29 pin) | ~281 | ~**35%** WEAK | pin was CPU0 |
| Hard+FIFO CCD1 quiet core (setcap, **no isolcpus**) | ~280–282 | ~**9–20%** WEAK | **product host path** |

**Host policy (2026-08-01): no boot `isolcpus`.** Product claims are size-fit + mlock + hard+FIFO tok/s; DRAM% stays **WEAK** on this desktop and is not a ship blocker.

Hard+FIFO is the preferred single-stream product path once `cap_sys_nice` is set.
**Pin policy:** `--cpu N` / `CLM_PIN_CPU` / taskset / quiet physical core of `--ccd` (avoid CPU0; prefer CCD1). Measure: `uv run python scripts/measure_residency.py`.

**setcap note:** `sudo setcap cap_sys_nice+ep c_infer/ll3m_c` after `make` if the binary was relinked.

Full ship freeze write-up: [SHIP_FREEZE.md](SHIP_FREEZE.md).

## Multi-session serve under the same CCD gate

Shared ternary weights + independent states:

```text
WS(N, kv_cap) = weights + N × state(kv_cap)
refuse if WS > 0.95 × one CCD L3
```

| Config | Max N | Aggregate (measured) |
|--------|------:|---------------------:|
| Full T=256 | **2** | ~550 tok/s |
| kv_cap=128 | **8** | ~2100 tok/s |
| kv_cap=64 | **8** | ~2190 tok/s (best balanced) |
| kv_cap=32–48 | **16** | ~2300 tok/s peak (SMT; ~145/user) |

```bash
make -C c_infer multi
./c_infer/multi_bench -m checkpoints/v76_dense_c/best.ll3m \
  --streams 8 --kv-cap 64 --ccd 1 -n 256
```

Full sweep: `checkpoints/compare/l3_isolation_bench/N_SESSION_MAX.md`.

Optional single-thread RR scheduler (research / fairness, not product agg):
```bash
./c_infer/multi_bench -m checkpoints/v76_dense_c/best.ll3m \
  --streams 8 --kv-cap 64 --ccd 1 -n 128 --burst 16
```
Measured (N=8 kv=64): pthread aggregate **~1843** tok/s; RR burst 1/16/64 **~214–218**
(single-core by design). Burst length is flat when weights are shared and L3-resident.
See [READAHEAD_DESIGN.md](READAHEAD_DESIGN.md) §10; `outputs/readahead-r1r2r4-results.json`.

## Readahead (L2 tiles + multi KV)

Structure-known software prefetch and multi-session KV scheduling (not learned
DRAM block correlation) are sketched in **[READAHEAD_DESIGN.md](READAHEAD_DESIGN.md)**.
Product path stays CCD size-fit + pin; readahead is L1/L2 tile hints and which
stream’s KV stays warm under `N × kv_cap`.
