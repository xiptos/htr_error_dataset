"""Shared plotting style for the EDA and results notebooks.

A small, colour-blind-validated categorical + sequential palette, applied
consistently across every figure produced in this repository.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK_SEC, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
seq_cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def apply_style() -> None:
    """Call once per notebook/script, before creating any figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SEC, "text.color": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.8,
        "font.family": "sans-serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def style_ax(ax) -> None:
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)


def save_fig(fig, name: str, out_dir: Path) -> None:
    """Save a figure as both vector PDF (for LaTeX/print) and 300dpi PNG (for quick preview)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
