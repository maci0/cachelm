#!/usr/bin/env bash
# Soft CPU isolation without reboot (best-effort).
# Picks a core, moves default IRQs off it, optional exclusive cpuset.
#
# Usage:
#   sudo scripts/residency_soft_isolate.sh [CPU]
#   sudo scripts/residency_soft_isolate.sh 7
#   scripts/residency_soft_isolate.sh status
#
set -euo pipefail

CPU="${1:-}"
NPROC=$(nproc)

if [[ "${CPU}" == "status" || ( -z "${CPU}" && "${1:-}" == "status" ) ]]; then
  echo "isolated sysfs: $(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo n/a)"
  echo "online: $(cat /sys/devices/system/cpu/online)"
  echo "default_smp_affinity: $(cat /proc/irq/default_smp_affinity 2>/dev/null || echo n/a)"
  if systemctl is-active --quiet irqbalance 2>/dev/null; then
    echo "irqbalance: active (may undo soft IRQ steer)"
  else
    echo "irqbalance: inactive/absent"
  fi
  echo "quiet-cpu hint: $(bash "$(dirname "$0")/residency_host_setup.sh" quiet-cpu 2>/dev/null || echo $((NPROC - 1)))"
  echo "see: scripts/residency_host_setup.sh protocol"
  exit 0
fi

if [[ -z "$CPU" ]]; then
  # Prefer CCD-aware quiet core from host_setup when available
  if [[ -x "$(dirname "$0")/residency_host_setup.sh" ]]; then
    CPU="$(bash "$(dirname "$0")/residency_host_setup.sh" quiet-cpu)"
  else
    CPU=$((NPROC - 1))
  fi
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "need root (sudo)" >&2
  exit 1
fi

if [[ "$CPU" -lt 0 || "$CPU" -ge "$NPROC" ]]; then
  echo "CPU $CPU out of range 0..$((NPROC-1))" >&2
  exit 1
fi

echo "soft-isolating CPU $CPU of $NPROC"

# Build mask for all CPUs except $CPU (hex affinity mask)
# smp_affinity is a bitmask of CPUs
python3 - "$CPU" "$NPROC" <<'PY'
import sys
cpu = int(sys.argv[1])
n = int(sys.argv[2])
mask = 0
for i in range(n):
    if i != cpu:
        mask |= 1 << i
# print as hex without 0x, comma groups of 8 for large masks
h = f"{mask:x}"
print(h)
print(f"0-{n-1} without {cpu} -> mask {h}", file=sys.stderr)
PY
MASK=$(python3 - "$CPU" "$NPROC" <<'PY'
import sys
cpu=int(sys.argv[1]); n=int(sys.argv[2])
mask=0
for i in range(n):
    if i!=cpu: mask|=1<<i
print(f"{mask:x}")
PY
)

# Move IRQs that have smp_affinity writable
moved=0
failed=0
for d in /proc/irq/*; do
  [[ -d "$d" ]] || continue
  af="$d/smp_affinity"
  [[ -w "$af" ]] || continue
  # skip irq 0
  base=$(basename "$d")
  [[ "$base" =~ ^[0-9]+$ ]] || continue
  if echo "$MASK" >"$af" 2>/dev/null; then
    moved=$((moved + 1))
  else
    failed=$((failed + 1))
  fi
done
echo "  IRQ affinity: moved=$moved failed=$failed  (others stay managed by irqbalance)"

# Stop irqbalance if present so it does not undo affinities
if systemctl is-active --quiet irqbalance 2>/dev/null; then
  systemctl stop irqbalance
  echo "  stopped irqbalance (restart later: systemctl start irqbalance)"
fi

# Default IRQ affinity for new IRQs
if [[ -w /proc/irq/default_smp_affinity ]]; then
  echo "$MASK" >/proc/irq/default_smp_affinity 2>/dev/null || true
  echo "  default_smp_affinity=$MASK"
fi

# Raise memlock for root sessions (process still needs ulimit)
# Write limits.d so user maci gets unlimited memlock
cat >/etc/security/limits.d/99-ll3m-residency.conf <<EOF
# LL3M residency: allow mlockall for hard mode
maci    soft    memlock    unlimited
maci    hard    memlock    unlimited
root    soft    memlock    unlimited
root    hard    memlock    unlimited
*       soft    memlock    unlimited
*       hard    memlock    unlimited
EOF
echo "  wrote /etc/security/limits.d/99-ll3m-residency.conf (new logins get unlimited memlock)"

# Persist hugepages across soft reboot via sysctl (best-effort)
if [[ -d /sys/kernel/mm/hugepages/hugepages-2048kB ]]; then
  cur=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)
  if [[ "$cur" -lt 64 ]]; then
    echo 64 >/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    echo "  nr_hugepages -> 64"
  else
    echo "  nr_hugepages already $cur"
  fi
  cat >/etc/sysctl.d/99-ll3m-hugepages.conf <<EOF
# LL3M: reserve 2MiB huge pages for MAP_HUGETLB model arena
vm.nr_hugepages = 64
EOF
  echo "  wrote /etc/sysctl.d/99-ll3m-hugepages.conf"
fi

echo
echo "done. Run inference pinned to CPU $CPU:"
echo "  ulimit -l unlimited"
echo "  taskset -c $CPU ./c_infer/ll3m_c -m MODEL --residency-hard --ccd 0 --greedy -n 200"
echo
echo "For full isolcpus/nohz (survives reboot), see:"
echo "  scripts/residency_host_setup.sh cmdline-hint"
