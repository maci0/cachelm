#!/usr/bin/env bash
# Max-param under L3 at T=128 (freeze lifted). ~99M ternary+int8.
# Default: L28 x d576 MQA (balanced width/depth). Override SHAPE=l36d512.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-checkpoints/v77_t128_l28d576}"
SHAPE="${SHAPE:-l28d576}"
STEPS="${STEPS:-40000}"
DATA="${DATA:-data/cap_mix_v1.txt}"
TOK="${TOK:-data/tokenizer.json}"

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "need ROCm torch" >&2
  exit 1
fi

case "$SHAPE" in
  l28d576) L=28; D=576; H=12; KV=1 ;;
  l36d512) L=36; D=512; H=16; KV=1 ;;
  l23d640) L=23; D=640; H=16; KV=1 ;;
  *) echo "unknown SHAPE=$SHAPE" >&2; exit 1 ;;
esac

mkdir -p "$OUT"
cat > "$OUT/RECIPE.txt" <<R
v77 denser under L3 (T=128)  shape=$SHAPE
L=$L d=$D h=$H kv=$KV T=128 V=2048 ternary+int8 from step 0 QAT
data=$DATA  steps=$STEPS
freeze lifted 2026-08-01; product gate still ~30.4 MiB at T=128
R

# sanity gate
"$PY" - <<PY
from ll3m.config import GPTConfig
from ll3m.model_torch import estimate_infer_ws
from ll3m.cache_info import probe_cache
cfg = GPTConfig(n_layer=$L, n_embd=$D, n_head=$H, n_kv_head=$KV, block_size=128,
                vocab_size=2048, peri_ln=False, layer_scale_init=1e-4)
gate = int(probe_cache().l3_bytes * 0.95)
inf = estimate_infer_ws(cfg, quant="ternary", embed_quant="int8")
print(f"params {cfg.n_params():,}  WS {inf.total/1024**2:.2f} / {gate/1024**2:.2f} MiB")
assert inf.total <= gate, "over gate"
assert cfg.n_params() == sum(p.numel() for p in __import__('ll3m.model_torch', fromlist=['GPT']).GPT(cfg).parameters())
print("gate OK")
PY

exec "$PY" -u -m ll3m.train \
  --arch v2 --tokenizer-mode bpe \
  --n-layer "$L" --n-embd "$D" --n-head "$H" --n-kv-head "$KV" --block-size 128 \
  --layer-scale-init 1e-4 \
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
  --batch-size 4 --grad-accum 16 \
  --dropout 0.1 --lr 3e-4 --warmup 2000 --steps "$STEPS" --patience 0 \
  --device rocm --infer-scope ccd --serve-quant ternary \
  --qat ternary --embed-qat int8 \
  --data "$DATA" --tokenizer "$TOK" \
  --prompt "Once upon a time" \
  --out-dir "$OUT"
