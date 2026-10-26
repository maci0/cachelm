#!/usr/bin/env bash
# Max-param under L3 with binary Linear pack (research reinvest).
# Shape: L30×d576 MQA mid2304 T=256 V=2048 ≈ 142.3M params
# WS binary+int8 ≈ 30.22 / 30.40 MiB (see docs/L3_CAPACITY_SEARCH.md)
#
# Native binary STE from step 0 (same philosophy as native ternary).
# Product ship remains ternary L3T2 until metrics promote.
#
#   bash scripts/train_dense_binary_l30.sh
#   OUT=checkpoints/v77_binary_l30 STEPS=50000 bash scripts/train_dense_binary_l30.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-checkpoints/v77_binary_l30}"
STEPS="${STEPS:-50000}"
DATA="${DATA:-data/mix_train.bin}"
TOK="${TOK:-data/tokenizer.json}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-2000}"
BS="${BS:-4}"
ACCUM="${ACCUM:-16}"

if ! "$PY" -c "import torch; assert torch.cuda.is_available(), torch.__version__" 2>/dev/null; then
  echo "need ROCm torch (got CUDA-only or CPU). Restore:" >&2
  echo "  uv pip install --python .venv/bin/python 'torch==2.10.0+rocm7.0' \\" >&2
  echo "    --index-url https://download.pytorch.org/whl/rocm7.0 --index-strategy unsafe-best-match" >&2
  exit 1
fi

mkdir -p "$OUT"
cat > "$OUT/RECIPE.txt" <<R
v77 binary denser under L3 (research reinvest)
L30 d576 h12/kv1 mid2304 T=256 V=2048
params ~142.3M  serve-quant binary + int8 embed
native --qat binary from step 0
data=$DATA  steps=$STEPS  lr=$LR
product ship stays ternary until promote
docs: docs/L3_CAPACITY_SEARCH.md
R

"$PY" - <<PY
from ll3m.config import GPTConfig
from ll3m.model_torch import estimate_infer_ws, GPT
from ll3m.cache_info import probe_cache
cfg = GPTConfig(
    vocab_size=2048, block_size=256, n_layer=30, n_head=12, n_embd=576,
    n_kv_head=1, intermediate_size=2304, peri_ln=False, layer_scale_init=1e-4,
)
gate = int(probe_cache().l3_bytes * 0.95)
inf = estimate_infer_ws(cfg, quant="binary", embed_quant="int8")
assert sum(p.numel() for p in GPT(cfg).parameters()) == cfg.n_params()
assert inf.total <= gate, (inf.total, gate)
print(f"params {cfg.n_params():,}  WS {inf.total/1024**2:.2f}/{gate/1024**2:.2f} MiB  OK")
PY

exec "$PY" -u -m ll3m.train \
  --n-layer 30 --n-embd 576 --n-head 12 --n-kv-head 1 \
  --intermediate-size 2304 --block-size 256 \
  --layer-scale-init 1e-4 \
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
  --batch-size "$BS" --grad-accum "$ACCUM" \
  --dropout 0.1 --lr "$LR" --warmup "$WARMUP" --steps "$STEPS" --patience 0 \
  --device rocm --infer-scope ccd \
  --serve-quant binary \
  --qat binary --embed-qat int8 \
  --data "$DATA" --tokenizer "$TOK" \
  --prompt "Once upon a time" \
  --out-dir "$OUT"
