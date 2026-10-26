"""Probe CPU cache hierarchy from sysfs (Linux)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheLevel:
    level: int
    type: str  # Data, Instruction, Unified
    size_bytes: int
    shared_cpu_count: int
    instances: int  # approximate chip-wide instance count

    @property
    def size_kib(self) -> float:
        return self.size_bytes / 1024

    @property
    def size_mib(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass(frozen=True)
class CacheHierarchy:
    levels: list[CacheLevel]
    l1d_bytes: int
    l1i_bytes: int
    l2_bytes: int  # per instance (usually per core)
    l3_bytes: int  # per instance (CCD on Zen)
    l3_total_bytes: int  # sum across instances
    n_cpus: int

    def summary(self) -> str:
        lines = [
            f"CPUs online: {self.n_cpus}",
            f"L1d (per core): {self.l1d_bytes / 1024:.0f} KiB",
            f"L1i (per core): {self.l1i_bytes / 1024:.0f} KiB",
            f"L2  (per core): {self.l2_bytes / 1024:.0f} KiB",
            f"L3  (per instance): {self.l3_bytes / (1024 * 1024):.1f} MiB",
            f"L3  (chip total):  {self.l3_total_bytes / (1024 * 1024):.1f} MiB",
        ]
        return "\n".join(lines)


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("K"):
        return int(s[:-1]) * 1024
    if s.endswith("M"):
        return int(s[:-1]) * 1024 * 1024
    if s.endswith("G"):
        return int(s[:-1]) * 1024 * 1024 * 1024
    return int(s)


def probe_cache() -> CacheHierarchy:
    """Read /sys/devices/system/cpu/cpu*/cache for hierarchy."""
    cpu_root = Path("/sys/devices/system/cpu")
    cpu_dirs = sorted(cpu_root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:]))
    n_cpus = len(cpu_dirs)
    if n_cpus == 0:
        # Fallback guesses (desktop-ish)
        return CacheHierarchy(
            levels=[],
            l1d_bytes=48 * 1024,
            l1i_bytes=32 * 1024,
            l2_bytes=1024 * 1024,
            l3_bytes=32 * 1024 * 1024,
            l3_total_bytes=64 * 1024 * 1024,
            n_cpus=1,
        )

    # Collect unique (level, type, size, shared_count) from cpu0, then count instances.
    by_key: dict[tuple[int, str, int, int], int] = {}
    per_cpu0: dict[tuple[int, str], CacheLevel] = {}

    for cpu in cpu_dirs:
        cache_dir = cpu / "cache"
        if not cache_dir.is_dir():
            continue
        for idx in sorted(cache_dir.glob("index*")):
            try:
                level = int((idx / "level").read_text().strip())
                ctype = (idx / "type").read_text().strip()
                size = _parse_size((idx / "size").read_text())
                shared = (idx / "shared_cpu_list").read_text().strip()
                # shared_cpu_list like "0-7,16-23" or "0,16"
                n_shared = _count_cpu_list(shared)
            except (OSError, ValueError):
                continue
            key = (level, ctype, size, n_shared)
            by_key[key] = by_key.get(key, 0) + 1
            if cpu.name == "cpu0":
                per_cpu0[(level, ctype)] = CacheLevel(
                    level=level,
                    type=ctype,
                    size_bytes=size,
                    shared_cpu_count=n_shared,
                    instances=1,
                )

    # instances = how many distinct caches of that size exist (cpus reporting / sharers)
    levels: list[CacheLevel] = []
    for (level, ctype, size, n_shared), reports in sorted(by_key.items()):
        instances = max(1, reports // max(1, n_shared))
        levels.append(
            CacheLevel(
                level=level,
                type=ctype,
                size_bytes=size,
                shared_cpu_count=n_shared,
                instances=instances,
            )
        )

    def _size(level: int, ctype: str, default: int) -> int:
        cl = per_cpu0.get((level, ctype))
        return cl.size_bytes if cl else default

    l1d = _size(1, "Data", 32 * 1024)
    l1i = _size(1, "Instruction", 32 * 1024)
    l2 = _size(2, "Unified", 512 * 1024)
    l3_level = per_cpu0.get((3, "Unified"))
    l3 = l3_level.size_bytes if l3_level else 16 * 1024 * 1024
    l3_instances = 1
    for cl in levels:
        if cl.level == 3 and cl.type == "Unified":
            l3_instances = cl.instances
            break
    l3_total = l3 * l3_instances

    return CacheHierarchy(
        levels=levels,
        l1d_bytes=l1d,
        l1i_bytes=l1i,
        l2_bytes=l2,
        l3_bytes=l3,
        l3_total_bytes=l3_total,
        n_cpus=n_cpus,
    )


def _count_cpu_list(s: str) -> int:
    n = 0
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            n += int(b) - int(a) + 1
        else:
            n += 1
    return n


def budget_bytes(
    hier: CacheHierarchy,
    fraction: float = 0.90,
    scope: str = "chip",
) -> int:
    """
    Training working-set budget.

    scope:
      - "ccd": one L3 instance (strictest Zen residency; pin threads to one CCD)
      - "chip": sum of all L3 (all cache on this system; default for max capacity)

    fraction leaves a little room for code, stack, and allocator slop.
    """
    base = hier.l3_total_bytes if scope == "chip" else hier.l3_bytes
    return max(256 * 1024, int(base * fraction))
