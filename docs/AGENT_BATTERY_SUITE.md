# Closed agent battery suite (101 cases)

**Status:** product evaluation card for LL3M (not BFCL / public When2Call).  
**Code:** `scripts/eval_agent_battery.py` · `ll3m/agent.py`  
**Ship pins:** `checkpoints/paper_freeze_v072/` (SHA256 in `MANIFEST.json`)

## Purpose

Measure **tool policy + host interaction** under explicit flags:

| Policy | `assist_tools` | `gate_tools` | Meaning |
|--------|----------------|--------------|---------|
| **Product** | off | **on** | Ship claim = model emission × When2Call-style host gate |
| **Pure** | off | **off** | Model-only stress (no host block/rewrite) |
| Assist / gold | on | either | Bootstrap ablation only (not ship) |

Product quality is a **system** score. Pure is the model-only number.

## Case inventory (101 total)

| Category | n | Source | What it checks |
|----------|--:|--------|----------------|
| `host` | 3 | Host unit (no model) | `host_assist_call` intent: chitchat vs tool |
| `no_call` | 24 | `build_agent_cases()` | Greetings / chitchat; must not **execute** tools |
| `calc` | 30 | same | Arithmetic via `calc`; expect answer substr |
| `lookup` | 20 | same | Closed KB facts via `lookup` |
| `time` | 5 | same | `time` tool |
| `adversarial` | 6 | same | Glued args, no-tool story, etc. |
| `language` | 12 | language openers | Free-form completion (not tool protocol) |
| `tokenizer` | 1 | pack encode check | BPE pack matches product tokenizer |

Agent protocol rows: **85**. Plus host 3 + language 12 + tokenizer 1 = **101**.

Exact prompts live in `build_agent_cases()` and the language list in `main()` of `eval_agent_battery.py`. This doc is the public schema; the code is the ground truth.

## Wire format

Model may emit:

```text
<call>tool_name arg1 arg2</call>
```

Host executes (`ll3m/tools.py`) and injects results or phrases a short answer.  
Product gate does **not** invent a call when the model is silent.

## Gate instrumentation (`gate_stats`)

Battery reports aggregate host-gate events from `run_agent` / `AgentTurn.gate_events`:

| Field | Meaning |
|-------|---------|
| `n_model_emitted_call` | Model produced a parseable `<call>` |
| `n_false_call_emitted` | Call when host intent is no-tool |
| `n_false_call_blocked` | Host blocked that false call |
| `n_tool_rewrite` | Host rewrote wrong tool name to gold intent |
| `n_true_call_emitted` | Call when a tool was needed |
| `n_silent_when_tool_needed` | Tool needed; model emitted no call |

Per-case fields: `gate_events`, `model_emitted_call`, `gold_needs_tool`, `executed_tools`.

**Pass/fail** for product uses **executed** tools + final text.  
**Gate stats** report what the model tried before the host.

## How to run

```bash
# Product (assist off, gate on)
uv run python scripts/eval_agent_battery.py \
  --checkpoint checkpoints/v76_dense_c_pure_v2/best.ll3m \
  --no-assist \
  --out checkpoints/v76_dense_c_pure_v2/battery_product_gate_on_gatestats.json

# Pure (assist off, gate off)
uv run python scripts/eval_agent_battery.py \
  --checkpoint checkpoints/v76_dense_c_pure_v2/best.ll3m \
  --no-assist --no-gate \
  --out checkpoints/v76_dense_c_pure_v2/battery_true_pure_greedy_gatestats.json
```

## Ship numbers (paper freeze)

| Policy | Score | Pin |
|--------|------:|-----|
| Product | **101/101** | `paper_freeze_v072/battery_product_gate_on_101.json` |
| Product (prior) | **99/101** | `…/battery_product_gate_on_99.json` |
| Pure | **90/101** | `…/battery_true_pure_greedy_90.json` |

Re-runs with gate instrumentation write new JSON with `gate_stats` (do not silently overwrite freeze pins).

## Mapping to public benchmarks

| Public suite | Relationship |
|--------------|--------------|
| BFCL / Gorilla | **Not** this suite |
| When2Call | **Inspiration only** for no-call / gate policy |
| Optional future | Map a subset of no_call + calc for external reporting |

## Limitations

1. Closed prompts; need this repo for exact cases.  
2. Tool-heavy; language openers are coarse.  
3. Product 101/101 is a **system** score when `gate_tools=True`.  
4. Single seed / greedy for pure-policy tool emission.  
5. KB-dependent lookup; freeze pins a KB snapshot hash.

## Related

- `docs/TOOL_CALLING.md`  
- `docs/paper/LL3M_L3_Resident_Ternary_SLM.md`  
- `outputs/ll3m-paper-v076-review.md`  


## Measured gate decomposition (pure_v2, 2026-08-03)

Re-run with instrumentation (`battery_*_gatestats.json`). Do **not** overwrite freeze pins.

### Product (assist off, gate on) — **101/101**

| Gate stat | Value |
|-----------|------:|
| Agent protocol cases | 85 |
| Model emitted `<call>` | **85/85** |
| Gold needs tool | 58 |
| True call emitted | **58/58** (0 silent) |
| False call emitted | **27** |
| False call **blocked** | **27/27** |
| Tool name rewrite | **2** |

Interpretation: the model tries a tool on every protocol case, including 27 no-tool intents. The host gate is what keeps product no_call at 24/24.

### Pure (assist off, gate off)

| Scoring | Score | Meaning |
|---------|------:|---------|
| Freeze pin (stream-visible) | **90/101** | Historical ship pin only — **not** regenerated by emission-aware or user-visible scorers |
| Emission-aware (executed false tool fails) | **72/101** | Model tool spam counted |
| User-visible final text only | **99/101** | Chitchat recovery hides many emissions |

Same 27 false-call emissions; **0 blocked**. Time fails 2/5 (lookup instead of time) under both pure scorings.

### Artifacts

- `checkpoints/v76_dense_c_pure_v2/battery_product_gate_on_gatestats.json`
- `checkpoints/v76_dense_c_pure_v2/battery_true_pure_greedy_gatestats.json`
- Unit tests: `tests/test_agent_gate_stats.py` (**5+** cases: gate block/rewrite, set_kb live alias, dual-score schema)
- Status JSON: `outputs/p1-eval-status.json`


## KB on/off ablation (lookup, pure_v2)

| Condition | Product gate | Pure |
|-----------|-------------:|-----:|
| KB on (~3k flat keys) | **20/20** | **20/20** |
| KB off (empty) | **0/20** | **0/20** |

Artifact: `checkpoints/v76_dense_c_pure_v2/kb_onoff_ablation.json`  
Script: `scripts/kb_onoff_ablation.py` (`ll3m.tools.set_kb`).

Lookup accuracy is **host-table dependent**; empty KB cannot be rescued by weights or the product gate.

## Related product docs

- [PRODUCT_CANDIDATE.md](PRODUCT_CANDIDATE.md) — ship packs + gate decomp table
- [TOOL_CALLING.md](TOOL_CALLING.md) — protocol + gate events
- [SHIP_FREEZE.md](SHIP_FREEZE.md) — freeze pins
- Paper **v0.7.9** §agent gate decomposition
