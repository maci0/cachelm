# Bare-metal C inference

Full project docs: [../README.md](../README.md) · [../docs/L3_RESIDENCY.md](../docs/L3_RESIDENCY.md) ·
[../docs/QUANTIZATION.md](../docs/QUANTIZATION.md) · [../docs/ARCHITECTURE_V2.md](../docs/ARCHITECTURE_V2.md).

## Role

Python/PyTorch trains the model. **Only this C runtime** is allowed to claim:

> Inference working set fits in **one L3 CCD** and stays on-chip after pin/warm
> (DRAM share of L2 fills ≈ noise).

No Python, no PyTorch on the hot path.

## Formats

| Magic | Model | C support |
|-------|--------|-----------|
| **CLM1** | v1 char GPT-2-style | **Full** (load → quant → forward → sample) |
| **CLM2** | v2 + tokenizer blob | Export ready; C forward **pending** |

Export:

```bash
# from repo root
.venv/bin/python scripts/export_weights.py \
  --checkpoint checkpoints/best_L6d192_infer.pt \
  --out checkpoints/best_L6.clm
```

## Quantization (at load)

| Build | Linear weights |
|-------|----------------|
| **Default** `-DCLM_QUANT=2` | **Ternary** `{-1,0,1}`, 2-bit packed, absmean scale |
| `-DCLM_QUANT=1` | int8 absmax (+ VNNI when available) |

Product L3T2 pack: **int8 embeds** + f32 norms; Linears ternary. KV and activations stay float32. (f32 embeds only in non-product L3M2/debug packs.)

Example (L6d192): ternary weights ~**0.9 MiB**, total infer WS ~**3.4 MiB**.

## L3 residency (required for the claim)

```bash
cd c_infer
make
ulimit -l unlimited   # if mlock fails

./ll3m_c -m ../checkpoints/best_L6.clm \
  -p $'ROMEO:\n' -n 200 --greedy \
  --l3-resident --warm 12 --ccd 0

# Hardware proof (WS fit + DRAM share ≤ 5%)
make cachestat
DRAM_MAX_PCT=2 ./measure_cache.sh ../checkpoints/best_L6.clm 200
```

### Cache hierarchy + stats (built-in)

| Tier | Role in decode |
|------|----------------|
| **L1i** | Hot kernels (GEMV / decode / RMSNorm in `.text.hot`) |
| **L1d** | One-token residual + work vectors (dedicated mlock pad) |
| **L2** | Attn scores + software-prefetched weight tiles |
| **L3** | Full weights + KV + prefill scratch |

Prefetch knobs (compile-time, `ll3m_v2.c`): `CLM_W_PREFETCH_AHEAD` (default 32),
`CLM_W_PREFETCH_HINT` (0=NTA / 1=T1 / 2=T0), `CLM_NEXT_LAYER_PREFETCH` (default 1).
Grid: `bash scripts/prefetch_grid.sh`. Design: [../docs/READAHEAD_DESIGN.md](../docs/READAHEAD_DESIGN.md).

Every run prints:

1. **Hierarchy plan** — L1d/L1i/L2/L3 sizes and what belongs in each  
2. **Hot pad fit** — L1d/L2 pad fill % vs cache capacity  
3. **L3 occupancy** — full WS vs one CCD  
4. **RAM residency** — `mincore` on model + state + hot pads  
5. **PMU over decode** — cache hit/miss, L1d, AMD L2 fills (local L3 vs DRAM)

Host helpers: `scripts/residency_host_setup.sh status|hugepages|cmdline-hint`.  
Disable PMU with `--no-cache-stats`.

### Build optimizations (L1i / throughput)

Default `make` uses LTO, section GC, 64B function align, hot/cold sections.

```bash
make -C c_infer -j          # optimized release
make -C c_infer size        # text size + sections
make -C c_infer pgo MODEL=../checkpoints/best_L6.clm PGO_N=200   # PGO
```

### What `--l3-resident` does

1. Bind threads to **one L3 CCD**  
2. **mlock** model + state arenas (also default at alloc)  
3. **Prefault** every cache line  
4. **Warm** full forwards so L3 holds the set  
5. **Exit 2** if infer WS > CCD L3  

See [../docs/L3_RESIDENCY.md](../docs/L3_RESIDENCY.md).

### Example proof (this host)

```
infer WS:  3.35 MiB / 32 MiB CCD
mlock:     locked
CCD bind:  OK
L2 fills:  ~97% local L3, ~3% DRAM
L3_RESIDENCY_GATE: PASS
```

## Build

```bash
make                    # ternary quant + AVX-512 (product default)
# int8 instead:
make clean && make CFLAGS='-O3 -march=native -mavx512f -mfma -std=c11 -Wall -DCLM_QUANT=1'
# disable mlock (not for residency claims):
make CFLAGS='... -DCLM_NO_MLOCK'
# prefetch knobs (optional; defaults are product):
make CFLAGS='... -DCLM_W_PREFETCH_AHEAD=64 -DCLM_W_PREFETCH_HINT=1'
```

### Multi-stream (same CCD)

```bash
make -C c_infer multi
./c_infer/multi_bench -m ../checkpoints/v76_dense_c/best.ll3m --streams 2 --ccd 1 -n 256
# single-thread RR burst (fairness; lower aggregate than pthread):
./c_infer/multi_bench -m ../checkpoints/v76_dense_c/best.ll3m --streams 8 --kv-cap 64 --burst 16
```

Shared weights + N states; refuses if `weights + N*state > 0.95×CCD`.
On this host: N=2 full T fits (~28.4 MiB WS); N=8 kv_cap=64 ~25.2 MiB; product multi is **pthread**.
`--burst K` is single-thread RR (fairness / research; aggregate ~1/N of pthread).

### Secondary: static link and/or embedded weights

Default product path still uses **external** `-m model.ll3m` (small binary, swap packs without rebuild).

| Target | What you get |
|--------|----------------|
| `make` | Dynamic link, external pack |
| `make static` | Fully static binary, still needs `-m` |
| `make embed EMBED_MODEL=path/to.ll3m` | Dynamic link + pack in `.rodata` (~file size + code) |
| `make static-embed EMBED_MODEL=...` | **Single-file ship**: static libc + embedded pack |

```bash
# one-file demo (~22 MiB for denser-C L3T2 + static libc)
make -C c_infer static-embed \
  EMBED_MODEL=../checkpoints/v76_dense_c/best.ll3m

# omit -m (or pass -m :embedded:)
./c_infer/ll3m_c --token-ids 12,34,56 -n 32 --greedy --l3-resident
```

Notes:

- Embed uses `objcopy -I binary` → symbols `_binary_clm_pack_ll3m_{start,end}`.
- Loader still **copies** the pack into the mlock arena (same residency path as file load).
- Embedded blob lives in the ELF image; working set math is still arena + state + pads.
- Static needs host static libc (`glibc-static` / musl toolchain). If link fails, use `embed` only.
- `make clean` drops `clm_pack.ll3m` / `clm_pack.o` intermediates.

## CLI flags

| Flag | Meaning |
|------|---------|
| `-m PATH` | Model `.ll3m` (or `:embedded:` when built with `embed` / `static-embed`) |
| *(omit `-m`)* | Uses embedded pack if present |
| `-p STR` | Prompt (legacy; prefer `--token-ids` from Python BPE) |
| `-n N` | New tokens |
| `--token-ids ID,ID,...` | Prefill token ids |
| `--greedy` | Argmax (temp=0) |
| `--bench` | Timed forwards |
| **`--l3-resident`** | CCD pin + pin + warm + refuse oversize |
| `--residency-hard` | mlockall + 1-core + FIFO + hugepage hints |
| `--ccd N` | Which L3 group (default 0) |
| `--warm N` | Warm full forwards |
| `--steady-warm N` | Decode-only steps before PMU |

## Layout

| File | Role |
|------|------|
| `ll3m.h` / `ll3m.c` | Load, quant, forward, sample, pin/warm/CCD |
| `main.c` | CLI |
| `measure_cache.sh` | perf + **L3_RESIDENCY_GATE** |
| `Makefile` | Build / cachestat |

## Why not Python for the claim?

Python RSS includes interpreter, allocator, and framework. Even tiny tensors do
not make the **process** L3-resident. C keeps **weights + acts** in two mlock’d
regions you can size and measure.

## Quality note

Post-hoc quant without QAT degrades samples more (especially ternary). Prefer
`--qat ternary` or `--qat int8` fine-tune before caring about greedy quality in C.
