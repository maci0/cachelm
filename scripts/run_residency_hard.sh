#!/usr/bin/env bash
# Run ll3m_c with best-effort residency host policy for this machine.
# Prefers quiet CPU, unlimited memlock, hard residency flags.
#
# Usage:
#   scripts/run_residency_hard.sh -m checkpoints/v76_dense_c_pure_v2/best.ll3m -n 256 --greedy
#   LL3M_CPU=15 scripts/run_residency_hard.sh -m MODEL.ll3m -n 256 --greedy
#   sudo scripts/run_residency_hard.sh -m MODEL.ll3m -n 256 --greedy   # mlockall / FIFO OK
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/c_infer/ll3m_c"

# Quiet CPU: LL3M_CPU env, else host_setup default (CCD-aware), else last CPU.
if [[ -n "${LL3M_CPU:-}" ]]; then
  CPU="$LL3M_CPU"
elif [[ -x "$ROOT/scripts/residency_host_setup.sh" ]]; then
  CPU="$("$ROOT/scripts/residency_host_setup.sh" quiet-cpu)"
else
  CPU=$(($(nproc) - 1))
fi

# Prefer quieter CCD on dual-CCD hosts (9950X: CCD1 = 8-15,24-31 → index 1)
CCD="${LL3M_CCD:-1}"

if [[ ! -x "$BIN" ]]; then
  make -C "$ROOT/c_infer" -j
  echo "note: rebuild clears setcap; re-run: sudo setcap cap_sys_nice+ep $BIN" >&2
fi

# Ensure hugepages if root and none reserved
if [[ "$(id -u)" -eq 0 ]]; then
  free=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages 2>/dev/null || echo 0)
  if [[ "${free:-0}" -lt 16 ]]; then
    echo 64 >/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || true
  fi
  ulimit -l unlimited || true
fi

ulimit -l unlimited 2>/dev/null || true

# Default pack if user only passes flags (no -m)
if [[ $# -eq 0 ]]; then
  set -- -m "$ROOT/checkpoints/v76_dense_c_pure_v2/best.ll3m" -n 256 --greedy
fi

echo "run: cpu=$CPU ccd=$CCD memlock=$(ulimit -l) hugefree=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages 2>/dev/null || echo ?)"
if command -v getcap >/dev/null 2>&1; then
  caps=$(getcap "$BIN" 2>/dev/null || true)
  echo "caps: ${caps:-none}"
  if [[ -z "$caps" || "$caps" != *cap_sys_nice* ]]; then
    echo "warn: no cap_sys_nice on ll3m_c → SCHED_FIFO may fail; sudo setcap cap_sys_nice+ep $BIN" >&2
  fi
fi
# --cpu keeps pin after engine rebind; taskset is belt-and-suspenders
exec taskset -c "$CPU" "$BIN" --residency-hard --ccd "$CCD" --cpu "$CPU" "$@"