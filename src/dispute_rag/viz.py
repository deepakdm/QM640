"""Shared figure style for report charts (static PNG, light surface, print-friendly).

Colors follow a validated categorical order (blue, orange, aqua, yellow, ...), assigned to
entities in a fixed order and never cycled. Single-series charts use slot 1. Grids and axes are
recessive; text stays in neutral ink rather than series color.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# Report conditions always wear the same color, whichever subset is plotted.
CONDITION_COLORS = {"C0": "#8a8984", "A": SERIES[0], "B": SERIES[1], "C": SERIES[2]}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def finish(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
