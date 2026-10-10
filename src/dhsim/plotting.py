"""Shared chart style: light surface, hairline grid, thin marks, colour by series identity.

Figures are built with the object-oriented API (``matplotlib.figure.Figure``), so nothing here
touches pyplot's global state or needs a GUI backend.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# Categorical order validated for colour-vision deficiency: slots 1-3 stay distinguishable
# in every pairing, and each chart also separates series by marker shape.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlepad": 10,
    "axes.labelsize": 9.5,
    "axes.labelcolor": INK_2,
    "axes.prop_cycle": cycler(color=SERIES),
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "text.color": INK,
    "font.size": 9.5,
    "font.family": "sans-serif",
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.markersize": 6.5,
    "lines.markeredgecolor": SURFACE,
    "lines.markeredgewidth": 1.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
    "legend.handlelength": 1.6,
    "errorbar.capsize": 0,
    "axes.unicode_minus": True,
}


@contextmanager
def style():
    with mpl.rc_context(RC):
        yield


def figure(nrows: int = 1, ncols: int = 1, *, width: float = 12.0, height: float = 4.2,
           **subplot_kw) -> tuple[Figure, np.ndarray]:
    """A Figure with an array of axes (always 2-D indexable via ``axes.flat``)."""
    fig = Figure(figsize=(width, height), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, **subplot_kw)
    return fig, axes


def save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with style():
        fig.savefig(path)
    return path


def compact(x: float, _pos=None) -> str:
    """Tick formatter: 1,500 -> 1.5K; keeps a proper minus sign."""
    sign = "−" if x < 0 else ""
    a = abs(x)
    if a >= 1e6:
        s = f"{a / 1e6:.3g}M"
    elif a >= 1e3:
        s = f"{a / 1e3:.3g}K"
    else:
        s = f"{a:.3g}"
    return sign + s


def money_axis(ax, axis: str = "y") -> None:
    fmt = FuncFormatter(compact)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def note(ax, text: str, loc: str = "upper right") -> None:
    """Small secondary-ink annotation in a corner of the axes."""
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5, color=INK_2,
            linespacing=1.4)


def end_label(ax, x: float, y: float, text: str, dx: float = 4, dy: float = 0) -> None:
    """Direct label at a line end, in secondary ink (text never wears the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center",
                fontsize=8.5, color=INK_2)


def headline(fig: Figure, title: str) -> None:
    """Figure-level headline: the one-sentence takeaway, left-aligned."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
