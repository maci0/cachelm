# Tool calling for ll3m

How a **~75–80M, L3-resident** model can grow useful tool use without
pretending it is GPT-scale open-domain agents.

## Implemented (code)

| Piece | Path |
|-------|------|
| Tool registry | `ll3m/tools.py` — `calc`, `time`, `lookup`, `wiki`, `note` |
| Closed KB (Engram-lite) | `ll3m/kb_facts.json` **v2**: entries + aliases + `flat` map (~1164 keys) |
| Lookup match | Word-bounded longest key (no `pi` ⊂ `pizza`); see `tool_lookup` |
| Agent loop | `ll3m/agent.py` — parse `<call>`, execute, inject `<result>`; `gate_tools` |
| Chat UI | `python -m ll3m.chat chat --mode agent` (default) |
| Synthetic SFT | `scripts/gen_tool_sft.py` → `data/tool_sft.txt` (When2Call no-call mix) |
| Fact Q&A SFT | `scripts/gen_fact_qa_sft.py` from KB flat |
| Tool train | `scripts/train_tool_sft.sh` — keep `--qat ternary --embed-qat int8` |
| Dense ladder | [DENSE_KNOWLEDGE_LADDER.md](DENSE_KNOWLEDGE_LADDER.md) (spark1 A→D) |

**Product defaults (2026-07-29):** `assist_tools=False`, `gate_tools=True`.

| Flag | Behavior |
|------|----------|
| `gate_tools` | Block tools on chitchat; rewrite **wrong tool name** when host intent is known (e.g. `lookup` → `time` for “what day is it”). Does not invent a call if the model is silent. |
| `assist_tools` | Full inject when model emits nothing / calc arg repair. Off for product honesty. |
| `--pure-policy` | assist off + gate off (stress test). |

Product closed battery on pure_v2 gate-on: **101/101** ([SHIP_FREEZE.md](SHIP_FREEZE.md)).
Gate event logging: `AgentTurn.gate_events` / battery `gate_stats` — see [AGENT_BATTERY_SUITE.md](AGENT_BATTERY_SUITE.md).  
Instrumented pure_v2 product: false-call **blocked 27/27**, rewrites **2** → **101/101** system score.  
Pure dual scores: emission-aware **72/101**, user-visible final **99/101**, freeze stream pin **90/101**.  
KB on/off (lookup n=20): **20/20 → 0/20** empty host table (`scripts/kb_onoff_ablation.py`).

## Reality check

| Fact | Implication |
|------|-------------|
| Model is story-trained, tiny | It will **not** invent reliable free-form JSON for arbitrary APIs |
| Context T ≈ 256 | Tool schemas must be **short**; few tools active at once |
| L3 / C serve | Tool **execution** is host-side; model only emits **calls** |
| Quality of SLM tool use (literature) | Fine-tune + special tokens + constrained decode ≫ prompt-only |

Research and practice (Octopus-style function tokens, in-vehicle SLM tool
calling, Microsoft SLM function-calling finetunes) all agree: for small models,
**narrow tool sets + synthetic SFT + host validation** win. Free-form
OpenAI-style tool JSON in the prompt alone does not.

---

## Design principle

Split responsibilities:

```
┌─────────────────────────────────────────────────────────┐
│  Host agent loop (Python or C wrapper)                  │
│  - tool registry, schemas, execution, retries           │
│  - constrained decoding / parse / validate              │
│  - builds next prompt with tool results                 │
└───────────────────────────┬─────────────────────────────┘
                            │ text in / text out
┌───────────────────────────▼─────────────────────────────┐
│  ll3m (weights in L3)                                │
│  - decide: reply | call tool | done                     │
│  - emit short structured call string                    │
└─────────────────────────────────────────────────────────┘
```

The model never “runs” tools. It only predicts tokens that the host maps to
`tool_name + args`.

---

## Recommended protocol: fixed registry + special tokens

### Why not full JSON schemas in the prompt?

At V=2048 and T=256, stuffing OpenAPI-style JSON wastes context and confuses a
story model. Prefer a **closed tool table** the model saw in training.

### Wire format (v0, human-readable, C-friendly)

```
User: what is 12 times 7?
Assistant: <call>calc mul 12 7</call>
Tool: <result>84</result>
Assistant: twelve times seven is 84.
```

Rules:

| Piece | Meaning |
|-------|---------|
| `<call>…</call>` | Model wants a tool; host must run it |
| name | First token after `<call>`: tool id from registry |
| args | Space-separated (or `key=value` for optional kwargs) |
| `<result>…</result>` | Host-injected; model never invents this in train as free text without a preceding call |
| plain text | Final user-facing answer (no call tags) |

### Special tokens (Octopus / in-vehicle SLM style)

Add to BPE / reserved ids (do **not** split them):

```
<call>  </call>  <result>  </result>  <tools>  </tools>
```

Optional: one token per tool for ultra-small sets:

```
<tool_calc> <tool_time> <tool_note> ...
```

Then a call is:

```
<tool_calc> mul 12 7</call>
```

Fewer free-form name errors; best when the tool set is **fixed at train time**.

### Keep the registry tiny (v0)

Start with 3–8 tools the story-scale model can actually learn:

| Tool | Args | Example |
|------|------|---------|
| `calc` | `op a b` | `add 2 3`, `mul 12 7` |
| `time` | (none) | wall clock string |
| `note` | free text | write a short memo |
| `story_len` | count words of last reply | meta demo |
| `lookup` | key in a **host dict** | `lookup capital france` (closed KB) |

Not in v0: web search, code sandbox, MCP kitchen-sink (too open, too long).

---

## Host agent loop

```
messages = [system_tools_brief, user]
loop up to N steps:
  out = model.generate(messages, stop_at=["</call>", "\nUser:"])
  if out contains <call>...</call>:
    parse → (name, args)
    if invalid: inject <result>ERROR: ...</result> and continue
    result = execute(name, args)   # host only
    messages += assistant_call + <result>result</result>
  else:
    return out as final answer
```

Caps:

- `max_tool_steps` (e.g. 3) so a tiny model cannot infinite-loop  
- `max_result_chars` so T=256 is not blown by tool spam  
- Always validate args with a schema before execute  

This loop lives in Python first (`ll3m/agent.py`); C can later do the same
with a fixed function table.

---

## Constrained decoding (the reliability multiplier)

Small models **fail format** more than they fail intent. Fix structure in the
sampler, not only in weights.

### Modes (increasing force)

1. **Stop strings** — stop at `</call>` or double newline (already partly in chat).  
2. **Tag-aware mask** — once `<call>` is emitted, only allow tool names from registry, then arg charset, then `</call>`.  
3. **Full grammar / FSM** — regex/CFG over the call line (like outlines / llama.cpp grammars).  

For ll3m C path, (2) is enough:

```
state: FREE | IN_CALL_NAME | IN_CALL_ARGS | IN_RESULT (host only)
```

FREE: normal vocab  
IN_CALL_NAME: only tokens that complete a registered tool name  
IN_CALL_ARGS: digits, letters, space, `=`, `-`, `.`  
After `</call>`: stop generation; host runs tool  

JSON schema constrained decoding is optional later; the line format above is
easier at V=2048.

---

## Existing public datasets (tool / function-calling traces)

Yes: there is a large open ecosystem. Most of it targets **1B–70B** chat models and
full JSON schemas, so for ll3m we **filter + rewrite** into our short protocol,
not train raw.

### High-value open sets

| Dataset | HF / link | Scale (order) | What it has | Fit for ll3m |
|---------|-----------|---------------|-------------|-----------------|
| **xLAM / APIGen** | [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60k | `query`, `tools` JSON, `answers` (name+args); **verified** (format + exec + semantic) | **Best seed**: single-turn, clean, ~21 domains. xLAM-**1B** already shows tiny models work on this style. Gate/login may apply. |
| **APIGen-MT** | [`Salesforce/APIGen-MT-5k`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | ~5k | Multi-turn function-calling trajectories | Good for multi-step loops; keep traces short for T=256 |
| **Glaive v2** | [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | ~100k+ | Chat + tools; includes **no-tool** chat | Strong for **when-not-to-call** balance |
| **ToolACE** | [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) | subset public; paper ~26k APIs | Single / parallel / dependent calls; non-tool cases | Rich; subset only if license OK; still rewrite format |
| **Hermes FC v1** | [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | tens of MB | Hermes `<tools>` / `<tool_call>` / structured JSON mode | Close to our tag style; often multi-tool JSON-heavy |
| **When2Call** | [`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) | ~15k SFT + prefs | **When** to call vs answer vs clarify; preference pairs | **Critical** for no-call / clarify behavior |
| **ToolMind** | [`Nanbeige/ToolMind`](https://huggingface.co/datasets/Nanbeige/ToolMind) | ~360k | Reasoning-enhanced tool trajectories + open-source mix | Large; mine simple single-tool slices only |
| **Hermes tool-use (UIGEN / smirki)** | e.g. `smirki/Tool-Calling-Dataset-UIGEN-X`, `interstellarninja/hermes_reasoning_tool_use` | varies | Reasoning + tool traces (ShareGPT-like) | Optional; filter hard for vocab/length |
| **DeepNLP Agent FC** | [`DeepNLP/Agent-Function-Calling-Open-Dataset`](https://huggingface.co/datasets/DeepNLP/Agent-Function-Calling-Open-Dataset) | growing | messages + tools + tool_call | Same rewrite path as xLAM |

### Related (not pure tool SFT, still useful)

| Resource | Use |
|----------|-----|
| **BFCL** (Berkeley Function-Calling Leaderboard) | Eval only, not train |
| **Tau-bench** | Multi-turn agent eval; hard for 20M |
| TRL SFT **tool-calling format** | Target schema for larger models; we map *into* our simpler line format |

### xLAM row shape (typical)

```json
{
  "query": "Find the sum of multiples of 3 and 5 between 1 and 1000...",
  "tools": "[{\"name\": \"math_toolkit.sum_of_multiples\", ...}]",
  "answers": "[{\"name\": \"math_toolkit.sum_of_multiples\", \"arguments\": {...}}]"
}
```

Rewrite target (ll3m v0):

```text
User: find the sum of multiples of 3 and 5 from 1 to 1000
Assistant: <call>sum_multiples lo=1 hi=1000 m=3,5</call>
```

(Only if that tool is in **our** closed registry, or we alias `math_toolkit.*` → a short name.)

### How to use them without drowning a 20M story model

```
public dataset
    → filter (len, vocab, #tools, single-call preferred)
    → map tool names → tiny registry (or drop row)
    → rewrite to <call>…</call> protocol
    → simplify English (optional rewrite pass)
    → mix with story base + When2Call-style no-call
    → SFT
```

**Filters that matter for us:**

| Filter | Why |
|--------|-----|
| Final text length ≲ 400–800 chars | T=256 BPE budget |
| ≤ 1–2 tools in the active list | Prompt fits |
| ≤ 1–2 parallel calls | Tiny models fail multi-tool JSON |
| Args are numbers / short strings | Not nested objects |
| Domain subset: math, time, lookup, weather-as-lookup | Align with host tools we implement |
| Drop code-heavy / multi-API enterprise tools | Out of vocab and scope |

**Balance:** roughly **1:1** call vs no-call (Glaive + When2Call + pure TinyStories). Too many call-only rows → model always emits `<call>`.

**Provenance note:** xLAM powered Salesforce’s **xLAM-1b-fc-r** (“tiny giant”) on BFCL. That is the closest public proof that **function-calling SFT works under ~1B**, still larger than us, so we stay narrower and more constrained.

### What we still synthesize ourselves

Public sets rarely cover:

- Host `<result>` then short English answer (full loop)  
- Error recovery (`ERROR: bad args`)  
- Exact special tokens / BPE of our tokenizer  
- Our 3–8 tool names only  

So: **mine structure and arg patterns from xLAM/Glaive**, **emit full multi-turn traces with a local simulator**.

---

## Training path (how the model learns tools)

Tool use is a **second stage** after story LM, not a random prompt at demo time.

### Stage 0 — base (current)

TinyStoriesV2-GPT4 + light Shakespeare. Narrative English, no tools.

### Stage 1 — format SFT (required)

Mix (~5–20% of tokens, or a dedicated short finetune):

1. **Single-call episodes** (from xLAM-filtered + templates)  
   User asks something only a tool can answer → assistant emits one call → result → short answer.  
2. **No-call episodes**  
   Pure chat / story + Glaive/When2Call-style irrelevance; model must **not** emit `<call>`.  
3. **Multi-step (rare)**  
   From APIGen-MT short traces or calc then answer.  
4. **Error recovery**  
   Host returns `ERROR: bad args` → model re-calls or apologizes (mostly synthetic).  

Generate / convert:

```python
# pseudo: public row → our protocol
row = xlam[i]
if too_long(row) or not mappable(row.tools): skip
name, args = simplify(row.answers[0])
text = f"User: {simplify_en(row.query)}\nAssistant: <call>{name} {fmt(args)}</call>\n"
# optional: run host mock → append Tool/Assistant turns
```

Keep language simple: short sentences, small vocab, few entities (TinyStories voice when rewriting).

### Stage 2 — optional tool-token expansion

If using per-tool tokens, extend tokenizer, resize `wte` (copy mean of old rows
for new rows), then SFT. Extra rows cost `n_new · d` floats in L3; keep
`n_new ≤ 16`.

### Stage 3 — optional QAT

Ternary/int8 QAT **after** tool SFT so serve quant does not erase the format.

### What not to do

- Expect zero-shot tool JSON from the story model.  
- Put 50 tools and OpenAPI blobs in a 256-token window.  
- Train only positive call examples (model will call tools always).  
- Let the model generate `<result>` without a host (hallucinated tool outcomes).

---

## Architecture changes: almost none

Tool calling is **not** a new attention type. Optional model-side knobs:

| Change | Needed? |
|--------|---------|
| New layers / MoE / tool head | **No** for v0 |
| Special tokens in vocab | **Yes** (or reserved multi-char strings that BPE keeps) |
| Separate “tool head” classifier | Optional later (pick tool id as classification); more C work |
| Longer `block_size` | Nice if multi-turn tool traces grow |
| System prefix with tool list | Yes, short: one line per tool |

If you ever add a **tool head**:

```
h_T → Linear → softmax over |tools|+1 (none)
```

then only emit args with the LM head when tool ≠ none. That helps tiny models
but splits the API; defer until text protocol plateaus.

---

## C / L3 implications

| Piece | Where it runs | In L3? |
|-------|---------------|--------|
| Weights + KV + acts | C infer | **Yes** (product claim) |
| Tool implementations | Host process | No (by design) |
| Registry + parser | Host (or small C side table) | Tiny; not model WS |
| Constrained logit mask | Sampler in C or Python | Code + small allowlists |

Product claim stays: **model working set in L3**. Tools are I/O, like a
keyboard.

---

## API sketch (Python)

```python
from ll3m.agent import ToolAgent, Tool

def calc(op: str, a: str, b: str) -> str:
    x, y = float(a), float(b)
    return str({"add": x+y, "mul": x*y, "sub": x-y, "div": x/y}[op])

agent = ToolAgent(
    checkpoint="checkpoints/v2_tools/best.pt",
    tools=[
        Tool("calc", "op a b", calc, help="arithmetic"),
        Tool("time", "", lambda: datetime.now().isoformat()),
    ],
    max_steps=3,
)
print(agent.run("What is 12 times 7?"))
```

Internals:

1. Build prompt with tool brief + user  
2. Generate with stop / mask  
3. Parse `<call>`  
4. Execute or error  
5. Append result; repeat  

---

## Evaluation (honest)

| Metric | How |
|--------|-----|
| **Call precision** | Among generations with `<call>`, fraction valid |
| **Call recall** | Among queries that need a tool, fraction that call |
| **Arg accuracy** | Parsed args match gold |
| **No-call accuracy** | Story/chat queries stay free of `<call>` |
| **End task** | Final answer correct after tool (calc especially) |
| **Format fail rate** | Broken tags / unknown tools |

Target for v0 (calc + time + note): format fail &lt; 5% with constrained decode;
arg accuracy high on in-distribution templates. Open-domain “use any API” is
out of scope.

---

## Phased plan

| Phase | Deliverable | Effort | Status |
|-------|-------------|--------|--------|
| **P0** | Spec + `docs/TOOL_CALLING.md` (this file) | small | **done** |
| **P1** | Synthetic tool dataset builder + mix into SFT | small | partial (`scripts/gen_tool_sft.py`, `data/tool_sft.txt`) |
| **P2** | `ll3m/agent.py` loop + parse/validate/execute | small | **done** (host assist bootstrap for weak policy) |
| **P3** | Constrained decode for `<call>…</call>` | medium | **partial** — `ll3m/constrained.py` (clip after `</call>`, repair incomplete open tags); agent default on (`constrain_tools=True`). C-side token mask still later |
| **P4** | Finetune so agent works **without** assist | medium | todo (research R-B: gold tool-SFT) |
| **P5** | Special tokens in tokenizer / export | medium | later |
| **P6** | C-side stop + optional name mask | later | later |
| **P7** | Optional tool classification head | only if needed | skip unless needed |

Do **not** block C L3T2 v2 serve on tools. Tools are a host feature first; weights
hold **language + tool policy**, not encyclopedic facts.

## Research cross-links

Frontier + arXiv findings that reinforce this design (full digests in
[RESEARCH_MODELS_2026.md](RESEARCH_MODELS_2026.md)):

| Source | Transfer |
|--------|----------|
| **DeepSeek Engram** arXiv:2601.07372 | Knowledge as O(1) external memory; host `lookup`/wiki ≈ Engram-lite |
| **Kimi K2** arXiv:2507.20534 | Industrial agentic synthetic data + verifiers |
| **Kimi K2.5** Agent Swarm | Parallel multi-tool orchestration (host-side fan-out) |
| **Nemotron / GLM-5** | Open agentic post-train recipes; long-horizon tools |
| **SLM tool papers** (2512.15943, 2504.19277, 2501.02342, 2603.00030) | Targeted SFT + special tokens + constrained decode beat “hope format works” |
| Sibling **GLM5.2 FINDINGS** | Retrieval-lift: tools ≫ parametric knowledge for tiny models |

Backlog IDs **R-B, R-E, R-H, R-I** in `AGENTS.md` track the next experiments.

---

## Suggested first tools (demo narrative)

Fits the TinyStories voice and still shows “agency”:

1. **calc** — “How many apples if 3 kids get 4 each?”  
2. **lookup** — closed map of story facts (`dragon_color=green`)  
3. **time** — “What day is it?” (host clock)  
4. **remember / recall** — host key-value (model emits `remember name=Lily`)  

That is already a complete demo: language → structured call → host effect →
language, without web or code execution.

---

## Lessons from `../GLM5.2` (toolsmith, 2026-07)

Local experiment: **toolsmith** on Qwen3.5-0.8B (same size class as our quality yardstick),
real sandboxed tools, retrieval-first thesis. Canonical write-ups:
`../GLM5.2/FINDINGS.md`, `RESULTS.md`, `docs/CAPABILITIES.md`, `docs/IMPROVEMENTS.md`.

### Thesis (proven there; product stance here)

| GLM5.2 finding | LL3M mapping |
|----------------|--------------|
| **Do not store knowledge in tiny weights; store retrieve/act skill** | Already product: Engram-lite host KB + tools; ternary weights = language + policy |
| North-star = **retrieval-lift** (tool − closed-book), not closed-book alone | Our MC gap vs Qwen is expected; agent no-assist + KB hit rate are the product metrics |
| Base Qwen 0.8B + **harness** beat every train recipe they tried at feasible scale | Harness quality (parse, retry, gate, compress results) can move more than more SFT |

Measured there (n=100, bootstrap CIs > 0): mean single-hop retrieval-lift **+0.43**; multi-hop still positive (**+0.21**) but lower absolute accuracy.

### Training paradoxes (do not repeat blindly)

| Finding | Implication for LL3M |
|---------|----------------------|
| **SFT improves format** (name_acc ↑) but **cuts agentic lift** (overwrites exploration) | Tool-only SFT D already **hurts** no-assist (37/101) and SciQ. Prefer **multi-mix** (dense CE replay) and early-stop on **agent** metrics, not train loss |
| **Multi-turn RL collapsed** base lift: 0.60 → 0.30 → 0.13 as steps grew | GRPO-lite only with **verifiers** and **small N**; kill runs if no-assist / lift falls. Do not “more RL” on weak reward |
| Logit-KD from 27B **≤ base** | Optional KD later; not first lever for tools |
| Bigger base (gemma 2B-class) **lost** to Qwen 0.8B on lift | Architecture/native tool habit matters more than param count for retrieval agents |

### Meta-skills to weight in data (from CAPABILITIES)

Train these harder than “more tools”:

1. **When not to call** (When2Call / no-call ~1:1)  
2. **Right tool + schema-valid args**  
3. **Failure recovery** (ERROR → reformulate, do not babble)  
4. **Abstain / “look it up”** when ungrounded (cardinal sin there: answer from memory when a tool was available)  
5. **Result compression** (only need for us when wiki/result is long; T=256 forces it)  
6. **Grounded answer** only from `<result>` (faithfulness)

Their reward sketch (`toolsmith/rewards.py`) is portable in spirit:

- format + schema + tool_match + arg_match  
- **−1 missed tool** when retrieval needed  
- **−0.5 false call** when no tool needed  
- faithfulness / abstention extras for multi-turn  
- **anneal format weight** early → 0 late (Search-R1 lesson)

### Harness > weights (high ROI for us)

GLM5.2 ranked **eval rigor + harness** above more GPU train:

| Steal for LL3M | Concrete |
|----------------|----------|
| **Retrieval-lift style metric** | closed-book fact Q vs same with `lookup` forced/available |
| **Contains / soft match**, not only exact string | battery already uses substr; keep it for free-form answers |
| **When2Call external** | no-call / over-call rate on pure policy |
| **Error recovery traces** | inject `ERROR: unknown tool` / bad args in SFT (D fails with `unknown tool 'the'`) |
| **Gold-verified traces only** | teacher paths that **actually** reach correct answer (not format-only imitation) |
| **Self-distill base’s own wins** | if C/MIX no-assist calc/lookup succeeds, SFT on those trajectories (on-policy), not foreign teacher style |
| **Do not grow tool surface** | they have 8 sandboxed tools; we stay **calc/time/lookup/wiki/note** until format is solid |

### What not to copy

| Skip | Why |
|------|-----|
| Live web search / Docker sandboxes as product claim | Outside L3 product; optional host later |
| 262K context story | We are T=256 by design |
| Multi-turn RL at 100+ steps | Proven collapse at 0.8B under weak reward |
| Format-only SFT as ship path | D is the cautionary tale |
| Claiming “SFT beats base” without agentic eval | Their static tool-eval looked good while SimpleQA lift fell |

### Ordered actions for LL3M (from this + our batteries)

1. **Ship metric = no-assist battery** (C already ~100/101; fix D with multi-mix, not more tool-only).  
2. **Verifiers** for GRPO-lite: calc exact, lookup hit, no-call correct, schema OK (port reward shape, not their web stack).  
3. **Error-recovery + no-call** rows in `gen_tool_sft` (anti over-call / anti `call the`).  
4. **Self-distill** successful no-assist C/MIX trajectories before teacher gold that assumes larger models.  
5. **Early-stop** tool SFT on no-assist pass rate + SciQ, never on CE alone.  
6. Optional later: KL-anchored format SFT if format is weak but agentic policy is strong.

---

## Bottom line

For ll3m, tool calling is:

1. A **tiny text protocol** (`<call>…</call>`)  
2. A **closed tool registry** trained with synthetic SFT  
3. A **host agent loop** with validation and results  
4. **Constrained decoding** so format does not depend on 70B-scale luck  

Not: larger backbone, MoE router, or full MCP at 20M params.

**GLM5.2 adds:** measure **retrieval-lift / no-assist**, treat **SFT and RL as dangerous** without CE replay and verifiers, invest in **harness + meta-skills**, self-distill **successful** trajectories.

Capability order stays:

> dense base LM → multi-mix tool SFT (CE replay) → no-assist eval → verifiers / GRPO-lite if needed → (optional) more tools
