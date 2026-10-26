# LL3M paper drafts

| File | Format | Notes |
|------|--------|--------|
| **`ll3m_l3_resident.pdf`** | **PDF** | **Built draft (figures + bibliography)** |
| `LL3M_L3_Resident_Ternary_SLM.md` | Markdown | Working draft **v0.7.9** (polish + audit fixes) |
| `ll3m_l3_resident.tex` | LaTeX | Submission-oriented article **v0.7.9** (sync with MD) |
| `ll3m_l3_resident.bib` | BibTeX | Expanded primary refs |
| `figures/` | PDF+PNG | Generated charts/diagrams (`gen_figures.py`) |
| `../../checkpoints/paper_freeze_v072/` | Pins | KB + product/pure agent batteries |

## v0.7.9 update summary (2026-08-03)

- KB on/off: lookup 20/20 → 0/20 (host memory, not weights)
- SciQ multi-seed probe limit 100: mean 60.7±1.2 (3 seeds)
- Gate decomposition retained from v0.7.8
- Related systems (not paper page yet): C prefetch knobs + multi `--burst` in [READAHEAD_DESIGN.md](../READAHEAD_DESIGN.md)

## v0.7.8 update summary (2026-08-03)

- P1 gate instrumentation: product **blocked 27/27** false calls; pure emission-aware **72/101**
- Suite doc `docs/AGENT_BATTERY_SUITE.md`; tests `tests/test_agent_gate_stats.py`
- Multi-seed MC helper script (not full n=400 grid yet)

## v0.7.7 update summary (2026-08-02)

- P0 framing from research review: pure-first agent metrics; product = model × host gate
- Claim→checkpoint map table; size-fit vs DRAM% honesty; “name and operationalize” novelty verb
- Absolute −12.8 pp MC gap lead; freeze pin path + MANIFEST SHA256

## v0.7.6 update summary (2026-07-30)

- Prose + page-budget pass (arXiv/MLSys single-column)
- Abstract/intro/discussion/conclusion tightened; dropped gap + C+ figures
- Venue notes appendix removed; **17 → 14 pages**

## v0.7.5 update summary (2026-07-30)

- Visual polish pass: figure KB counts 1022/3057; pie legend; residency title/footer
- Equation/int4 baseline reflow (no overfull hbox)
- Float barriers + `[H]` locks for arch/ladder/pyramid/residency/multi tables
- Pyramid bottom label untruncated; size-tradeoff label stagger

## v0.7.3 update summary (2026-07-30)

- Submission polish: tighter abstract/agent prose; less audit meta in body
- Agent table ships **101/101** product + **90/101** pure; prior 99/101 kept as footnote text only
- Layout: novelty/residency tables and int4 equation reflow; appendix artifact table uses wrap-friendly paths
- Reproducibility points at `paper_freeze_v072` pins

## v0.7.2 update summary (2026-07-30)

- Paper–code audit fixes: Engram **1022** entries / **3057** keys (was 895/2649)
- Product agent: current `battery_product_gate_on.json` **101/101**; prior same-flag **99/101** pinned separately
- Footer version string fixed (was stale v0.6)
- Artifact pins: `checkpoints/paper_freeze_v072/MANIFEST.json`

## v0.7.1 update summary (2026-07-28)

- TeX Pareto section: dual-ship implication + careful DPO P3 (pure 90→93/101; product pure_v2 then 99/101)
- Engram ship snapshot then **895** entries / **2649** keys (superseded in v0.7.2)
- Settled dual ship: pure_v2 product agent; dense_c_p1 SciQ peak
- Residency path: `scripts/residency_host_setup.sh protocol|check-caps|cmdline-hint`

## v0.7 update summary (2026-07-27)

- Hard+FIFO single-stream **~280–282 tok/s** preferred over soft (~260–270)
- GOOD DRAM% (≤5%) achievable with `cap_sys_nice` FIFO on long windows
- Multi-session under CCD gate: N=2 full T (~550 agg); N=8 kv=64–128 (~2100–2200); N=16 short KV peak (~2300)
- New table `tab:multi`; six budgets (incl. multi-session density)
- Artifacts: `c_infer/multi_bench`, `l3_isolation_bench/N_SESSION_MAX.md`

## v0.6 review-fix summary

- Fixed MD Qwen per-task column to portable int4 (was stale `vs_qwen4` vector)
- Canonical compare path: `checkpoints/compare/ll3m_vs_qwen35_family_int4.json` (now includes **C+**)
- Added measured C residency (size PASS, DRAM% WEAK) + figure
- Pack **triple**: design WS 30.22 / C payload 24.32 (weights payload 20.27) / on-disk `.ll3m` **20.32** MiB
- Agent tables use **full paths** (pure_v2 99/90; denser-C pure 42; D 37; MIX 70)
- Relative gap body+abstract **23.6%**; pure_v2 SciQ **57.75**; BoolQ C→C+ **−8.25**
- Pareto denominators; COPA n=100; KB ship (then 895/2649; see v0.7.2 for 1022/3057)
- Expanded `.bib`; related work; evaluation cards
- Quarantined conflicting `battery_C_no_assist.json`
- Dense mix: `data/base_en_dense/manifest.json` (161M tok; bins may be train-host only)
- Product test freeze: `tests/test_product_dense.py` = L23×d576 / 78,815,712

## Figures

| File | Content |
|------|---------|
| `fig_system_arch` | L3 decode + host Engram tools |
| `fig_ws_breakdown` | Design-time infer WS pie under 30.4 MiB gate |
| `fig_residency_pmu` | Measured size fit + PMU L2 fill sources |
| `fig_ladder` | Dense knowledge ladder A→MIX (with C+) |
| `fig_density_pyramid` | Knowledge density hierarchy |
| `fig_macro_bars` | Macro MC vs Qwen (incl. C+) |
| `fig_gap_relative` | Absolute / relative / param gap |
| `fig_size_tradeoff` | Accuracy vs params (bubble size = pack) |
| `fig_pertask` | Per-task C/C+/D/MIX vs Qwen |
| `fig_cplus_specialize` | C→C+ task deltas |

Regenerate:

```bash
.venv/bin/python docs/paper/figures/gen_figures.py
```

## Build PDF

```bash
export PATH="$HOME/.local/bin:$PATH"
cd docs/paper
tectonic -X compile ll3m_l3_resident.tex
# → ll3m_l3_resident.pdf

# or classic TeX Live
pdflatex ll3m_l3_resident && bibtex ll3m_l3_resident && pdflatex ll3m_l3_resident && pdflatex ll3m_l3_resident
```

## Retarget

| Venue | Action |
|-------|--------|
| **arXiv** | Build PDF as-is; fill authors; expand ethics/license |
| **MLSys** | Drop sections into MLSys template; anonymize; page limit cut; lead with residency |
| **ACL / EMNLP** | Port into `acl.sty` template; shorten systems residency section if needed |
| **Workshop** | Use shortened abstract + §§1,3,6,7 only |

## Qwen3.5 family baseline (portable weight-only int4)

Same MC protocol as LL3M:

| Model | Artifact |
|-------|-------------------|
| 0.8B / 2B / 4B / 9B Base | `checkpoints/compare/Qwen__Qwen3.5-*_int4_limit400.json` |
| Aggregate vs LL3M C/C+/D/MIX | `checkpoints/compare/ll3m_vs_qwen35_family_int4.json` |
| Residency logs | `checkpoints/compare/residency_{soft,hard}_v76_dense_c.txt` |

```bash
# on CUDA host (spark1)
LIMIT=400 bash scripts/run_qwen35_family_int4.sh
```

**Do not** use `checkpoints/dense_pipeline/vs_qwen4_*.json` for portable-int4 claims (stale vector / different macro).

## Before submission

- [x] Author: Marcel Wysocki (sole)  
- [x] MD/TeX Qwen per-task sync  
- [x] Expanded BibTeX  
- [x] D/MIX agent battery tables  
- [x] On-host residency measurement (honest WEAK DRAM%)  
- [x] Quarantine bad battery JSON  
- [ ] Affiliation / contact (optional)  
- [x] isolcpus-grade DRAM% **skipped on this host** (policy: no isolcpus; claim size-fit + mlock + FIFO)  
- [ ] Wall-clock GPU hours / energy  
- [ ] License / model card statement  
