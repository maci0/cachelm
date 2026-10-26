#!/usr/bin/env bash
# Host setup helpers for LL3M L3 residency measurements.
#
# These are *host* policies (kernel cmdline / huge pages / caps). The engine
# already does mlock, CCD/isolated CPU bind, hierarchical touch, and GEMV tiling.
#
# Usage:
#   scripts/residency_host_setup.sh status
#   scripts/residency_host_setup.sh topology
#   scripts/residency_host_setup.sh protocol
#   scripts/residency_host_setup.sh check-caps
#   scripts/residency_host_setup.sh cmdline-hint
#   sudo scripts/residency_host_setup.sh hugepages 64    # 64 × 2MiB = 128 MiB
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmd="${1:-status}"

# Prefer a quiet physical core: last CPU on CCD1 when 2×L3 (9950X: CCD1 = 8-15,24-31).
# Fall back to last online CPU.
default_quiet_cpu() {
  local last
  last=$(($(nproc) - 1))
  # Prefer CPU 15 (phys core on CCD1) when ≥16 logical CPUs (SMT: 15↔31).
  if [[ "$(nproc)" -ge 16 ]]; then
    echo 15
  else
    echo "$last"
  fi
}

topology() {
  echo "=== L3 CCD topology (shared_cpu_list) ==="
  declare -A seen=()
  local cpu list
  for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    [[ -d "$cpu/cache/index3" ]] || continue
    list=$(cat "$cpu/cache/index3/shared_cpu_list" 2>/dev/null || true)
    [[ -n "$list" ]] || continue
    if [[ -z "${seen[$list]+x}" ]]; then
      seen[$list]=1
      size=$(cat "$cpu/cache/index3/size" 2>/dev/null || echo "?")
      echo "  L3 slice $size  CPUs: $list"
    fi
  done
  if [[ ${#seen[@]} -eq 0 ]]; then
    echo "  (could not read L3 shared_cpu_list)"
  fi
  echo
  echo "  nproc=$(nproc)  online=$(cat /sys/devices/system/cpu/online 2>/dev/null || echo ?)"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E 'Model name|Socket|Core|Thread|L3' | sed 's/^/  /' || true
  fi
  echo
  echo "  Product gate: one CCD L3 × 0.95  (this host: 32 MiB CCD → ~30.4 MiB)"
  echo "  Suggested quiet CPU for hard pin: $(default_quiet_cpu)  (SMT sibling often +ncores)"
}

status() {
  echo "=== CPU isolation ==="
  if [[ -f /sys/devices/system/cpu/isolated ]]; then
    isol=$(cat /sys/devices/system/cpu/isolated)
    if [[ -n "${isol// }" ]]; then
      echo "  isolated CPUs: $isol"
    else
      echo "  isolated CPUs: (none)  ← boot isolcpus not set"
    fi
  else
    echo "  isolated: sysfs missing"
  fi
  echo "  cmdline isol/nohz/irq:"
  tr ' ' '\n' </proc/cmdline | grep -E 'isolcpus|nohz|irqaffinity|rcu_nocbs' \
    | sed 's/^/    /' || echo "    (no isol/nohz flags)"

  echo
  topology | head -n 20

  echo
  echo "=== hugepages (2 MiB) ==="
  grep -E 'HugePages_|Hugepagesize|Hugetlb' /proc/meminfo || true
  if [[ -d /sys/kernel/mm/hugepages/hugepages-2048kB ]]; then
    echo -n "  nr_hugepages: "; cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    echo -n "  free_hugepages: "; cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages
  fi
  echo
  echo "=== memlock ==="
  echo "  ulimit -l: $(ulimit -l)"
  echo
  echo "=== caps / FIFO readiness ==="
  check_caps_brief
  echo
  echo "=== tips ==="
  echo "  status/topology/protocol/check-caps/cmdline-hint as subcommands"
  echo "  hugepages (root):  $0 hugepages 64"
  echo "  soft isol (root):  sudo scripts/residency_soft_isolate.sh \$( $0 quiet-cpu )"
  echo "  measure product:   scripts/run_residency_hard.sh -m checkpoints/v76_dense_c_pure_v2/best.ll3m -n 256 --greedy"
}

check_caps_brief() {
  local bin="$ROOT/c_infer/ll3m_c"
  if [[ -x "$bin" ]] && command -v getcap >/dev/null 2>&1; then
    local caps
    caps=$(getcap "$bin" 2>/dev/null || true)
    if [[ -n "$caps" ]]; then
      echo "  getcap ll3m_c: $caps"
    else
      echo "  getcap ll3m_c: (none)  ← need cap_sys_nice for SCHED_FIFO without root"
    fi
  else
    echo "  getcap: build c_infer/ll3m_c and install libcap2-bin for FIFO check"
  fi
  if command -v sysctl >/dev/null 2>&1; then
    echo "  kernel.sched_rt_runtime_us=$(sysctl -n kernel.sched_rt_runtime_us 2>/dev/null || echo ?)"
  fi
}

check_caps() {
  echo "=== capability check for hard+FIFO residency ==="
  check_caps_brief
  echo
  local bin="$ROOT/c_infer/ll3m_c"
  if [[ ! -x "$bin" ]]; then
    echo "building ll3m_c..."
    make -C "$ROOT/c_infer" -j
  fi
  echo
  echo "Grant FIFO without full root (one-time, root):"
  echo "  sudo setcap cap_sys_nice+ep $bin"
  echo "  getcap $bin   # expect cap_sys_nice=ep"
  echo
  echo "Or run measure as root:"
  echo "  sudo scripts/run_residency_hard.sh -m PACK -n 256 --greedy"
  echo
  echo "GOOD DRAM% (≤5% L2 fills from DRAM) needs:"
  echo "  1) size-fit + mlock (engine)"
  echo "  2) single quiet core (taskset / isolcpus)"
  echo "  3) SCHED_FIFO (cap or root)"
  echo "  4) long steady window (--steady-warm, -n ≥128)"
  echo "  5) optional boot isolcpus for near-zero OS noise"
}

hugepages() {
  local n="${1:-64}"
  local path=/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
  if [[ ! -w "$path" ]] && [[ "$(id -u)" -ne 0 ]]; then
    echo "need root to write $path" >&2
    exit 1
  fi
  echo "$n" >"$path"
  echo "set nr_hugepages=$n (2 MiB each → $((n * 2)) MiB reserved)"
  grep -E 'HugePages_Total|HugePages_Free' /proc/meminfo
}

cmdline_hint() {
  local q
  q=$(default_quiet_cpu)
  # Build irqaffinity as all CPUs except quiet + SMT sibling when possible
  local ncores smt irq
  ncores=$(($(nproc) / 2))
  if [[ "$(nproc)" -ge 32 ]]; then
    # 9950X: physical 0-15, SMT 16-31. Quiet phys core $q; sibling $((q+16))
    # Isolate both siblings of one core on CCD1.
    local sib=$((q + ncores))
    cat <<EOF
Suggested kernel cmdline for this host (Ryzen-class dual CCD, nproc=$(nproc)):

  isolcpus=${q},${sib} nohz_full=${q},${sib} rcu_nocbs=${q},${sib} irqaffinity=0-14,16-30

Rationale (9950X-shaped):
  - CCD0 L3: CPUs 0-7,16-23
  - CCD1 L3: CPUs 8-15,24-31
  - Isolating phys core ${q} + SMT sibling ${sib} keeps one quiet core on CCD1
  - Product decode still uses --ccd matching that L3 slice (engine maps shared_cpu_list)

Shorter single-CPU isol (if you only want one logical CPU):

  isolcpus=${q} nohz_full=${q} rcu_nocbs=${q} irqaffinity=0-$((q - 1)),$((q + 1))-$(($(nproc) - 1))

Notes:
  - isolcpus: scheduler avoids those CPUs for normal tasks
  - nohz_full: fewer timer ticks
  - rcu_nocbs: RCU callbacks off those CPUs
  - irqaffinity: default IRQs elsewhere (verify /proc/irq/*/smp_affinity)

After reboot:
  cat /sys/devices/system/cpu/isolated   # should list ${q} (and sibling if set)
  bash $0 status
  ulimit -l unlimited
  LL3M_CPU=${q} scripts/run_residency_hard.sh \\
    -m checkpoints/v76_dense_c_pure_v2/best.ll3m -n 256 --greedy

Optional hugepages for MAP_HUGETLB model arena:
  sudo $0 hugepages 64

Without reboot (weaker):
  sudo scripts/residency_soft_isolate.sh ${q}
  sudo setcap cap_sys_nice+ep $ROOT/c_infer/ll3m_c
EOF
  else
    cat <<EOF
Suggested kernel cmdline (pick a core not used by desktop IRQs, e.g. ${q}):

  isolcpus=${q} nohz_full=${q} rcu_nocbs=${q}

After reboot:
  cat /sys/devices/system/cpu/isolated
  ulimit -l unlimited
  LL3M_CPU=${q} scripts/run_residency_hard.sh -m PACK -n 256 --greedy

Optional: sudo $0 hugepages 64
EOF
  fi
}

protocol() {
  cat <<'EOF'
# Path to GOOD DRAM% (≤5% of L2 fills from DRAM)

## Layers (do not skip)

| Layer | What | How |
|-------|------|-----|
| A. Size | WS ≤ one CCD | design gate 30.4 MiB; C runtime ~24.3 MiB |
| B. Pin | pages present | mlock / mlockall via --residency-hard |
| C. Quiet core | no desktop noise | boot isolcpus **or** soft IRQ steer + taskset |
| D. RT | fewer preemptions | SCHED_FIFO via cap_sys_nice or root |
| E. Steady PMU | after warm | --steady-warm (hard default 64) + -n ≥128 |
| F. Optional | hugepages | nr_hugepages ≥ 32 (2 MiB) for MAP_HUGETLB |

## Without reboot (current desktop path)

```bash
make -C c_infer -j
sudo setcap cap_sys_nice+ep c_infer/ll3m_c
sudo scripts/residency_soft_isolate.sh 15   # or: scripts/residency_host_setup.sh quiet-cpu
ulimit -l unlimited
LL3M_CPU=15 scripts/run_residency_hard.sh \
  -m checkpoints/v76_dense_c_pure_v2/best.ll3m -n 256 --greedy
# Built-in PMU section: look for DRAM share of L2 fills ≤5% → GOOD
```

Expect: size PASS, mlock 100%, tok/s ~280, DRAM GOOD on long windows when FIFO sticks.
Without cap/root FIFO or with busy IRQs: DRAM often WEAK (~20–40%).

## With boot isolcpus (near-zero OS noise)

1. `scripts/residency_host_setup.sh cmdline-hint` → paste into bootloader
2. Reboot; confirm `cat /sys/devices/system/cpu/isolated`
3. Same run_residency_hard.sh as above
4. Optional external: `DRAM_MAX_PCT=5 c_infer/measure_cache.sh PACK 256`

## Pass / fail language (honest)

| Claim | Condition |
|-------|-----------|
| Size-fit PASS | runtime WS < 32 MiB CCD |
| mlock PASS | mincore 100% model+state |
| DRAM GOOD | L2 fill DRAM share ≤5% after steady warm |
| DRAM WEAK | size OK but DRAM share high (desktop noise / no FIFO) |
| isolcpus-grade | boot isolated CPU + GOOD DRAM on repeated runs |

Do **not** claim absolute zero DRAM without isolcpus + quiet host + GOOD PMU.
Product freeze still uses design estimator (30.22 MiB) as architecture gate.

## Artifacts

- Protocol doc: docs/L3_RESIDENCY.md
- Multi-session: c_infer/multi_bench, checkpoints/compare/l3_isolation_bench/
- Paper tables: soft vs hard+FIFO; multi N under gate
EOF
}

quiet_cpu() {
  default_quiet_cpu
}

case "$cmd" in
  status) status ;;
  topology|topo) topology ;;
  protocol|dram-protocol|path) protocol ;;
  check-caps|caps) check_caps ;;
  hugepages) hugepages "${2:-64}" ;;
  cmdline-hint|isol-hint) cmdline_hint ;;
  quiet-cpu) quiet_cpu ;;
  *)
    echo "usage: $0 status|topology|protocol|check-caps|hugepages [N]|cmdline-hint|quiet-cpu" >&2
    exit 1
    ;;
esac
