#!/usr/bin/env python3
"""Generate publication figures for the LL3M paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

C_TEAL = "#0d9488"
C_TEAL_D = "#0f766e"
C_ORANGE = "#ea580c"
C_SLATE = "#334155"
C_GRAY = "#94a3b8"
C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#16a34a"
C_RED = "#dc2626"
C_BG = "#f8fafc"


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, facecolor="white", edgecolor="none")
        print(f"wrote {p}")
    plt.close(fig)


def fig_macro_bars() -> None:
    models = [
        "Qwen3.5\n0.8B int4",
        "LL3M C\ndense cont.",
        "LL3M C+\n+15k dense",
        "LL3M MIX\nmulti-mix",
        "LL3M D\ntool SFT",
        "LL3M prior\ntool path",
    ]
    acc = [54.21, 41.39, 41.36, 41.18, 40.46, 40.11]
    colors = [C_ORANGE, C_TEAL, "#14b8a6", C_TEAL_D, C_BLUE, C_GRAY]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(models))
    bars = ax.bar(x, acc, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
    ax.axhline(54.21, color=C_ORANGE, ls="--", lw=1.0, alpha=0.5)
    ax.set_ylabel("Macro accuracy (%)")
    ax.set_title("Closed-book MC macro (7 tasks, n=400) vs Qwen3.5-0.8B int4")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(C_BG)
    for b, v in zip(bars, acc):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 1.2,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=C_SLATE,
        )
    # gap annotation
    ax.annotate(
        "−12.8 pp\n(~24% rel.)",
        xy=(1, 41.39),
        xytext=(3.2, 52),
        arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.2),
        fontsize=9,
        color=C_SLATE,
        ha="center",
    )
    ax.annotate(
        "C+ macro flat\n(SciQ +3)",
        xy=(2, 41.36),
        xytext=(3.5, 58),
        arrowprops=dict(arrowstyle="->", color=C_TEAL_D, lw=1.0),
        fontsize=8,
        color=C_TEAL_D,
        ha="center",
    )
    save(fig, "fig_macro_bars")


def fig_pertask() -> None:
    tasks = ["HellaSwag", "ARC-E", "WinoG", "BoolQ", "OBQA", "SciQ", "COPA"]
    c = [28.0, 27.3, 47.5, 56.0, 13.5, 66.5, 51.0]
    cplus = [28.5, 29.8, 48.0, 47.8, 13.0, 69.5, 53.0]
    d = [26.5, 31.0, 51.0, 54.3, 16.0, 48.5, 56.0]
    mix = [29.0, 27.8, 50.0, 46.5, 12.5, 66.5, 56.0]
    qwen = [35.5, 59.0, 49.8, 63.0, 21.0, 84.3, 67.0]
    x = np.arange(len(tasks))
    w = 0.16

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.bar(x - 2 * w, c, w, label="LL3M C", color=C_TEAL)
    ax.bar(x - w, cplus, w, label="LL3M C+", color="#14b8a6")
    ax.bar(x, mix, w, label="LL3M MIX", color=C_TEAL_D)
    ax.bar(x + w, d, w, label="LL3M D", color=C_BLUE)
    ax.bar(x + 2 * w, qwen, w, label="Qwen 0.8B int4", color=C_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-task accuracy: denser ladder vs Qwen3.5-0.8B int4")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(C_BG)
    # highlight SciQ drop / C+ SciQ
    ax.annotate(
        "SciQ: D drops\nMIX recovers",
        xy=(5, 48.5),
        xytext=(3.2, 82),
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.1),
        fontsize=8,
        color=C_RED,
    )
    ax.annotate(
        "C+: SciQ +3\nBoolQ −8",
        xy=(5, 69.5),
        xytext=(5.8, 88),
        arrowprops=dict(arrowstyle="->", color=C_TEAL_D, lw=1.0),
        fontsize=8,
        color=C_TEAL_D,
    )
    save(fig, "fig_pertask")


def fig_cplus_specialize() -> None:
    """C → C+ task deltas: SciQ specialization without macro lift."""
    tasks = ["HellaSwag", "ARC-E", "WinoG", "BoolQ", "OBQA", "SciQ", "COPA", "Macro"]
    delta = [0.5, 2.5, 0.5, -8.25, -0.5, 3.0, 2.0, -0.04]
    colors = [C_TEAL if v >= 0 else C_RED for v in delta]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(tasks))
    ax.barh(y, delta, color=colors, height=0.65, edgecolor="white")
    ax.axvline(0, color=C_SLATE, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Accuracy change (pp): C+ − C")
    ax.set_title("Dense continue +15k under QAT: SciQ up, BoolQ down, macro flat")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(C_BG)
    for i, v in enumerate(delta):
        ax.text(v + (0.15 if v >= 0 else -0.15), i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=C_SLATE, fontweight="bold")
    ax.set_xlim(-10, 5)
    save(fig, "fig_cplus_specialize")


def fig_size_tradeoff() -> None:
    # bubble: params vs macro, size ~ pack
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # name, params_m, macro, pack_mib, color
    # (name, params_m, macro, pack_mib, color, text_dx_log, text_dy)
    # Stagger LL3M labels: all share ~79M params so default offsets collide.
    pts = [
        ("LL3M C", 79, 41.39, 21, C_TEAL, 1.35, 1.4),
        ("LL3M C+", 79, 41.36, 21, "#14b8a6", 1.35, 0.15),
        ("LL3M MIX", 79, 41.18, 21, C_TEAL_D, 1.35, -1.1),
        ("LL3M D", 79, 40.46, 21, C_BLUE, 1.35, -2.4),
        ("Qwen 0.8B\nint4", 750, 54.21, 400, C_ORANGE, 1.12, 0.7),
        ("Qwen 2B\nint4", 1880, 59.39, 900, "#fb923c", 1.12, 0.7),
        ("Qwen 9B\nint4", 8950, 69.11, 3500, "#ea580c", 0.55, -2.2),
    ]
    for name, p, m, pack, col, dx, dy in pts:
        s = min(pack, 800) * 0.35 + 40
        ax.scatter([p], [m], s=s, c=col, alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(p * dx, m + dy, name, fontsize=7, color=C_SLATE, va="center", ha="left")
    ax.axvline(79, color=C_TEAL, ls=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Macro accuracy (%)")
    ax.set_title("Accuracy vs scale (marker area ∝ approx. serve footprint)")
    ax.set_ylim(35, 75)
    ax.set_xlim(50, 15000)
    ax.set_facecolor(C_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for pack, lab in [(20.3, "LL3M L3T2 file ~20.3 MiB"), (400, "Qwen int4 pack (approx.)")]:
        ax.scatter([], [], s=min(pack, 200) * 0.5 + 40, c=C_GRAY, alpha=0.6, label=lab)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.text(
        0.02,
        0.02,
        "Qwen bubble sizes are schematic multi-GB class, not byte-exact dumps.",
        transform=ax.transAxes,
        fontsize=7,
        color=C_GRAY,
    )
    save(fig, "fig_size_tradeoff")


def fig_ws_breakdown() -> None:
    # Design-time conservative estimator (shape freeze). C runtime measures ~24.3 MiB.
    labels = [
        "Design weights (~21.2 MiB)",
        "Design acts+KV (~9.0 MiB)",
    ]
    sizes = [21.2, 9.0]
    colors = [C_TEAL, C_BLUE]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.72,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=8,
    )
    ax.set_title(
        "Design-time infer WS (~30.22 MiB of 30.40 MiB gate)\n"
        "C runtime payload ~24.3 MiB · ~0.18 MiB design headroom",
        fontsize=11,
    )
    centre = Circle((0, 0), 0.30, fc="white")
    ax.add_artist(centre)
    ax.text(0, 0.05, "L3\nCCD", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SLATE)
    ax.text(0, -0.12, "est. 30.2", ha="center", va="center", fontsize=8, color=C_GRAY)
    fig.tight_layout()
    save(fig, "fig_ws_breakdown")


def fig_residency_pmu() -> None:
    """Measured C decode residency on product host (not a PASS claim)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    # Left: size fit
    ax = axes[0]
    cats = ["L3T2 pack\nfile", "C runtime\nWS", "Design\nestimator", "CCD\n32 MiB", "Product\ngate 95%"]
    vals = [20.27, 24.32, 30.22, 32.0, 30.40]
    cols = [C_TEAL, C_TEAL_D, C_BLUE, C_GRAY, C_ORANGE]
    x = np.arange(len(cats))
    ax.bar(x, vals, color=cols, width=0.7, edgecolor="white")
    ax.axhline(30.40, color=C_ORANGE, ls="--", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("MiB")
    ax.set_ylim(0, 36)
    ax.set_title("Size fit (v76 denser-C L3T2)")
    ax.set_facecolor(C_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, color=C_SLATE, fontweight="bold")

    # Right: L2 fill sources (hard mode, no isolcpus)
    ax = axes[1]
    labels = ["Local L3\n(this CCD)", "DRAM\n(near+far)", "Other"]
    # hard run: L3 73.37%, DRAM 26.63%
    fracs = [73.37, 26.63, 0.0]
    colors = [C_TEAL, C_RED, C_GRAY]
    ax.bar(labels, fracs, color=colors, width=0.65, edgecolor="white")
    ax.set_ylabel("% of L2 fills")
    ax.set_ylim(0, 100)
    ax.set_title("PMU L2 fill sources (hard mode)\nmlock 100% · DRAM share WEAK")
    ax.set_facecolor(C_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(fracs):
        if v > 0:
            ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold", color=C_SLATE)
    # Host note moved to figure-level footer (avoids title collision).
    fig.suptitle(
        "Measured C residency (size PASS; DRAM% not yet target)",
        fontsize=11,
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Host: Ryzen-class 2×32 MiB L3 · no isolcpus · ~269 tok/s hard / ~294 tok/s soft",
        ha="center",
        fontsize=7.5,
        color=C_SLATE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, "fig_residency_pmu")


def fig_system_arch() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("white")

    def box(x, y, w, h, title, body, color, title_c="white"):
        r = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.95,
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10, fontweight="bold", color=title_c)
        ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", fontsize=8, color=title_c, linespacing=1.35)

    # Host plane
    host = FancyBboxPatch(
        (0.3, 0.3),
        9.4,
        5.4,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=C_BG,
        edgecolor=C_GRAY,
        linewidth=1.2,
        linestyle="--",
    )
    ax.add_patch(host)
    ax.text(0.5, 5.4, "Host process (DRAM free)", fontsize=9, color=C_SLATE, fontweight="bold")

    # L3 box
    l3 = FancyBboxPatch(
        (0.6, 1.6),
        4.6,
        3.4,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor="#ccfbf1",
        edgecolor=C_TEAL,
        linewidth=2.0,
    )
    ax.add_patch(l3)
    ax.text(2.9, 4.7, "One L3 CCD  (gate ≈ 30.4 MiB)", ha="center", fontsize=10, fontweight="bold", color=C_TEAL_D)

    box(0.9, 3.35, 4.0, 1.1, "LL3M decode (C, L3T2)", "79M · ternary Linear + int8 emb\nC WS ≈ 24.3 MiB · T=256", C_TEAL)
    box(0.9, 1.9, 1.85, 1.2, "L3T2 pack", "20.3 MiB file\n(~21.2 design)", C_TEAL_D)
    box(2.95, 1.9, 1.95, 1.2, "State + KV", "~4.0 MiB C arena\nMQA, short T", "#0891b2")

    # Host tools
    box(5.7, 3.5, 3.7, 1.5, "Engram-lite tools", "lookup (KB v2) · calc · time\nwiki (capped) · note", C_PURPLE)
    box(5.7, 1.9, 3.7, 1.3, "Closed KB", "1022 entries / 3057 keys\naliases · word-boundary", "#6d28d9")

    # User
    box(5.7, 0.55, 3.7, 1.1, "User / agent loop", "<call>…</call>  →  <result>", C_SLATE)

    # decode <-> tools
    ax.annotate(
        "",
        xy=(5.7, 4.2),
        xytext=(5.2, 4.2),
        arrowprops=dict(arrowstyle="<->", color=C_PURPLE, lw=1.6, mutation_scale=12),
    )
    # Label sits above the L3/tools gap so it is not clipped by either box.
    ax.text(5.45, 5.05, "calls / results", fontsize=7, color=C_PURPLE, ha="center", va="bottom", zorder=6)
    # tools -> user
    ax.annotate(
        "",
        xy=(7.55, 1.65),
        xytext=(7.55, 1.9),
        arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.4, mutation_scale=12),
    )
    ax.set_title("LL3M product split: L3-resident decode + host knowledge", fontsize=12, pad=8)
    save(fig, "fig_system_arch")


def fig_ladder() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        (0.15, "A", "Host KB v2", "Engram-lite\n1022 / 3057", C_PURPLE),
        (2.15, "B", "Dense data", "atomic + edu\nnear-dedup", C_BLUE),
        (4.15, "C", "Continue QAT", "+30k steps\nmacro 41.4%", C_TEAL),
        (6.15, "C+", "+15k dense", "SciQ 69.5%\nmacro flat", "#14b8a6"),
        (8.15, "D/MIX", "Tool / multi", "tools + CE\nreplay", C_ORANGE),
    ]
    for x, tag, title, body, col in stages:
        r = FancyBboxPatch(
            (x, 1.0),
            1.9,
            2.2,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=col,
            edgecolor="white",
            linewidth=2,
            alpha=0.92,
        )
        ax.add_patch(r)
        ax.text(x + 0.95, 2.85, tag, ha="center", fontsize=14, fontweight="bold", color="white")
        ax.text(x + 0.95, 2.35, title, ha="center", fontsize=9, fontweight="bold", color="white")
        ax.text(x + 0.95, 1.55, body, ha="center", fontsize=7.5, color="white", linespacing=1.3)
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 1.9
        x2 = stages[i + 1][0]
        ax.annotate(
            "",
            xy=(x2 - 0.05, 2.1),
            xytext=(x1 + 0.05, 2.1),
            arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.8, mutation_scale=14),
        )
    ax.text(6, 0.45, "QAT ternary + int8 locked on every train stage", ha="center", fontsize=9, color=C_SLATE, style="italic")
    ax.set_title("Dense knowledge ladder (product curriculum)", fontsize=12)
    save(fig, "fig_ladder")


def fig_density_pyramid() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    layers = [
        (1.0, 7.6, 8.0, 1.5, "Host KV / triples / tools", "highest density for product facts", C_PURPLE),
        (1.5, 5.8, 7.0, 1.5, "Atomic train cards / Q&A", "high info per token", C_TEAL),
        (2.0, 4.0, 6.0, 1.5, "Short edu prose", "language + light knowledge", C_BLUE),
        (2.5, 2.2, 5.0, 1.5, "Raw web pages", "low density, needed for fluency", C_ORANGE),
        (2.8, 0.35, 4.4, 1.55, "Rare facts in\nternary weights", "avoid (capacity fails)", C_GRAY),
    ]
    for x, y, w, h, title, sub, col in layers:
        r = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=col,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.2, color="white", alpha=0.95)
    ax.set_title("Knowledge density hierarchy (product bytes)", fontsize=12, pad=6)
    save(fig, "fig_density_pyramid")


def fig_gap_relative() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    # horizontal comparison
    cats = ["Absolute gap\n(percentage points)", "Relative deficit\n(1 − LL3M/Qwen)", "Params ratio\n(Qwen / LL3M)"]
    vals = [12.82, 23.6, 9.5]
    units = ["pp", "%", "×"]
    colors = [C_ORANGE, C_RED, C_BLUE]
    y = np.arange(len(cats))
    ax.barh(y, vals, color=colors, height=0.55, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("Value")
    ax.set_title("LL3M C vs Qwen3.5-0.8B weight-only int4 (macro MC)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(C_BG)
    for i, (v, u) in enumerate(zip(vals, units)):
        ax.text(v + 0.3, i, f"{v:.1f}{u}", va="center", fontsize=10, fontweight="bold", color=C_SLATE)
    ax.set_xlim(0, max(vals) * 1.25)
    save(fig, "fig_gap_relative")


def main() -> None:
    fig_macro_bars()
    fig_pertask()
    fig_cplus_specialize()
    fig_size_tradeoff()
    fig_ws_breakdown()
    fig_residency_pmu()
    fig_system_arch()
    fig_ladder()
    fig_density_pyramid()
    fig_gap_relative()
    print("all figures →", OUT)


if __name__ == "__main__":
    main()
