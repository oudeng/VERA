"""Critical-difference (CD) diagrams -- Demsar (2006) Figure 1 style.

A CD diagram is the standard visual companion to Friedman + Nemenyi: methods are
placed on an average-rank axis (best on the left), and methods whose average
ranks differ by less than the critical difference are joined by a horizontal
clique bar, meaning "not distinguishable at this sample size".

For the R1 response this figure carries most of the argument for R2-6a: with
n = 12 settings and k = 9 methods the Nemenyi CD is large, so most of the field
sits inside one clique.  That is not a defect of the plot -- it is the honest
statement of how little 12 paired settings can resolve.

Matplotlib is imported with the Agg backend so the module is safe on a headless
server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["find_cliques", "plot_cd_diagram"]


def find_cliques(avg_ranks: pd.Series, cd: float) -> List[Tuple[int, int]]:
    """Maximal groups of consecutive (rank-ordered) methods within ``cd``.

    Returns index pairs ``(i, j)`` into the rank-sorted ordering.  Groups fully
    contained in another group are dropped, and singletons are omitted.
    """
    order = avg_ranks.sort_values()
    vals = order.to_numpy(dtype=float)
    k = vals.size
    raw: List[Tuple[int, int]] = []
    for i in range(k):
        j = i
        while j + 1 < k and (vals[j + 1] - vals[i]) <= cd:
            j += 1
        if j > i:
            raw.append((i, j))
    cliques: List[Tuple[int, int]] = []
    for a in raw:
        if not any((b[0] <= a[0] and a[1] <= b[1] and b != a) for b in raw):
            if a not in cliques:
                cliques.append(a)
    return cliques


def plot_cd_diagram(
    avg_ranks: pd.Series,
    cd: float,
    *,
    title: str = "",
    subtitle: str = "",
    out_path: Optional[str | Path] = None,
    width: float = 9.0,
    row_height: float = 0.36,
    highlight: Optional[Sequence[str]] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Draw a Demsar critical-difference diagram.

    Parameters
    ----------
    avg_ranks
        Average ranks per method, lower = better.
    cd
        Critical difference (see :func:`stats.posthoc.nemenyi_critical_difference`).
    highlight
        Methods to draw in bold / accent color (typically ``["SNI"]``).
    out_path
        When given, the figure is written as PDF and PNG next to each other.
    """
    order = avg_ranks.sort_values()
    names = [str(x) for x in order.index]
    vals = order.to_numpy(dtype=float)
    k = len(names)
    highlight = set(highlight or [])

    lo = float(np.floor(min(vals) - 0.5))
    hi = float(np.ceil(max(vals) + 0.5))
    lo = max(1.0, lo)
    hi = min(float(k), max(hi, lo + 1.0))

    n_left = int(np.ceil(k / 2))
    n_right = k - n_left
    n_rows = max(n_left, n_right)
    height = 2.9 + row_height * n_rows
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(hi, lo)  # best (lowest rank) on the left
    ax.set_ylim(-(n_rows * row_height + 0.9), 1.75)
    ax.axis("off")

    axis_y = 0.0
    ax.plot([lo, hi], [axis_y, axis_y], color="black", lw=1.2, zorder=3)
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1e-9, 1.0)
    for t in ticks:
        ax.plot([t, t], [axis_y, axis_y + 0.07], color="black", lw=1.0, zorder=3)
        ax.text(t, axis_y + 0.12, f"{t:g}", ha="center", va="bottom", fontsize=9)

    # CD ruler, anchored at the best rank
    cd_y = axis_y + 0.50
    cd_left = lo
    cd_right = min(hi, lo + cd)
    ax.plot([cd_left, cd_right], [cd_y, cd_y], color="black", lw=1.4, zorder=3)
    for xx in (cd_left, cd_right):
        ax.plot([xx, xx], [cd_y - 0.06, cd_y + 0.06], color="black", lw=1.4, zorder=3)
    ax.text(
        (cd_left + cd_right) / 2.0, cd_y + 0.11, f"CD = {cd:.3f}",
        ha="center", va="bottom", fontsize=9,
    )

    # Method stems
    for i, (name, v) in enumerate(zip(names, vals)):
        left_side = i < n_left
        row = i if left_side else (k - 1 - i)
        y = axis_y - 0.30 - row * row_height
        edge = lo if left_side else hi
        bold = name in highlight
        color = "#b2182b" if bold else "black"
        lw = 1.7 if bold else 1.0
        ax.plot([v, v], [axis_y, y], color=color, lw=lw, zorder=2)
        ax.plot([v, edge], [y, y], color=color, lw=lw, zorder=2)
        label = f"{name}  ({v:.2f})"
        ax.text(
            edge, y, ("  " + label) if left_side else (label + "  "),
            ha="left" if left_side else "right",
            va="center", fontsize=10,
            fontweight="bold" if bold else "normal", color=color,
        )

    # Clique bars
    cliques = find_cliques(order, cd)
    for c_i, (i, j) in enumerate(cliques):
        y = axis_y - 0.10 - c_i * 0.075
        ax.plot(
            [vals[i] - 0.03, vals[j] + 0.03], [y, y],
            color="#2166ac", lw=3.2, solid_capstyle="butt", zorder=4,
        )

    if title:
        ax.text(
            (lo + hi) / 2.0, axis_y + 1.55, title,
            ha="center", va="center", fontsize=12, fontweight="bold",
        )
    if subtitle:
        ax.text(
            (lo + hi) / 2.0, axis_y + 1.20, subtitle,
            ha="center", va="center", fontsize=9, color="#444444",
        )

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if out_path.suffix.lower() != ".png":
            fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    return fig
