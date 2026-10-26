#!/usr/bin/env bash
# C-headroom binary reinvest (research). Spend measured CCD free bytes on params.
#
# Shape: L34×d576 MQA mid2880 T=256 V=2048 ≈ 194.9M params
# Estimator WS ~38 MiB (overstates act); C-calibrated total ~29.95 / 30.40 gate.
# Product ship remains ternary L3T2 until metrics promote.
#
#   bash scripts/train_dense_binary_c195.sh
#   OUT=checkpoints/v78_binary_c195 STEPS=60000 bash scripts/train_dense_binary_c195.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-checkpoints/v78_binary_c195}"
STEPS="${STEPS:-60000}"
DATA="${DATA:-data/mix_train.bin}"
TOK="${TOK:-data/tokenizer.json}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-2000}"
BS="${BS:-4}"
ACCUM="${ACCUM:-16}"
GPU_FRAC="${GPU_FRAC:-0.55}"

if ! "$PY" -c "import torch; assert torch.cuda.is_available(), torch.__version__" 2>/dev/null; then
  echo "need ROCm torch (got CUDA-only or CPU). Restore:" >&2
  echo "  uv pip install --python .venv/bin/python 'torch==2.10.0+rocm7.0' \\" >&2
  echo "    --index-url https://download.pytorch.org/whl/rocm7.0 --index-strategy unsafe-best-match" >&2
  exit 1
fi

mkdir -p "$OUT"
cat > "$OUT/RECIPE.txt" <<R
v78 binary C-headroom reinvest (research)
L34 d576 h12/kv1 mid2880 T=256 V=2048
params ~194.9M  serve-quant binary + int8 embed
native --qat binary from step 0
data=$DATA  steps=$STEPS  lr=$LR  bs=$BS accum=$ACCUM
Gate: C-calibrated WS ~29.95 MiB under product 30.40 (est overstates act).
Prior v77 L30×d576 mid2304 was 142.3M @ C 23.0 MiB; +7.4 MiB headroom spent here.
product ship stays ternary until promote
docs: docs/L3_CAPACITY_SEARCH.md
R

"$PY" - <<'PY'
from ll3m.config import GPTConfig
from ll3m.model_torch import estimate_infer_ws, GPT
from ll3m.cache_info import probe_cache

cfg = GPTConfig(
    vocab_size=2048, block_size=256, n_layer=34, n_head=12, n_embd=576,
    n_kv_head=1, intermediate_size=2880, peri_ln=False, layer_scale_init=1e-4,
)
gate = int(probe_cache().l3_bytes * 0.95)
inf = estimate_infer_ws(cfg, quant="binary", embed_quant="int8")
n = sum(p.numel() for p in GPT(cfg).parameters())
assert n == cfg.n_params(), (n, cfg.n_params())
# C smoke on v77: act scale ~0.388, weight ~0.958 → C-like total
c_like = (inf.weight_bytes * 0.9576 + inf.activation_bytes * 0.3882) / 1024**2
print(
    f"params {cfg.n_params():,}  estWS {inf.total/1024**2:.2f} MiB  "
    f"C-like ~{c_like:.2f} / {gate/1024**2:.2f} MiB gate"
)
assert c_like <= gate / 1024**2 + 0.05, (c_like, gate / 1024**2)
print("C-like gate OK (est may exceed 30.4; C is authority for residency)")
PY

# --force: estimator overstates act (~2.6× vs C); C-like ~29.95 ≤ 30.4 product gate.
exec "$PY" -u -m ll3m.train \
  --n-layer 34 --n-embd 576 --n-head 12 --n-kv-head 1 \
  --intermediate-size 2880 --block-size 256 \
  --layer-scale-init 1e-4 \
  --z-loss 1e-4 --mtp 0.1 --mtp-depth 2 \
  --batch-size "$BS" --grad-accum "$ACCUM" \
  --gpu-mem-fraction "$GPU_FRAC" \
  --dropout 0.1 --lr "$LR" --warmup "$WARMUP" --steps "$STEPS" --patience 0 \
  --device rocm --infer-scope ccd --force \
  --serve-quant binary \
  --qat binary --embed-qat int8 \
  --data "$DATA" --tokenizer "$TOK" \
  --prompt "Once upon a time" \
  --out-dir "$OUT"
