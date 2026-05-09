"""Plotting functions for comparative and multivariate models."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from drippy.data import EDAData

_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


def block_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a block plot of y vs treatment, grouped by block.

    Shows treatment effects within each block as connected line segments,
    one series per block level.

    Args:
        data: EDAData container. Requires factors with keys
            ``"treatment"`` and ``"block"``.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.

    Raises:
        ValueError: If factors is None or missing required keys.
    """
    if (
        not data.factors
        or "treatment" not in data.factors
        or ("block" not in data.factors)
    ):
        msg = "block_plot requires factors with 'treatment' and 'block' keys"
        raise ValueError(msg)
    treatment = data.factors["treatment"]
    block = data.factors["block"]
    fig, ax = get_figure_and_axes(fig, ax)
    for i, b in enumerate(np.unique(block)):
        mask = block == b
        t_vals = treatment[mask]
        y_vals = data.y[mask]
        sort_idx = np.argsort(t_vals)
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(
            t_vals[sort_idx],
            y_vals[sort_idx],
            marker=marker,
            linestyle="-",
            label=str(b),
        )
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Y")
    ax.set_title("Block Plot")
    ax.legend(title="Block")
    fig.tight_layout()
    return fig, ax


def youden_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    doe: bool = False,
) -> tuple[Figure, Axes]:
    """Creates a Youden plot comparing two labs or measurement methods.

    Plots Lab 1 (y) vs Lab 2 (x) with an equality line and median
    reference lines to reveal bias and lab effects.

    Args:
        data: EDAData container. Requires x (Lab 2 measurements)
            and y (Lab 1 measurements).
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        doe: If True, overlays DOE design point markers.

    Returns:
        The figure and axes containing the plot.

    Raises:
        ValueError: If x is None.
    """
    if data.x is None:
        msg = "youden_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    ax.scatter(data.x, data.y)
    lo = min(data.x.min(), data.y.min())
    hi = max(data.x.max(), data.y.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="y = x")
    med_y = np.median(data.y)
    med_x = np.median(data.x)
    ax.axhline(med_y, color="gray", linestyle=":", label="Median Lab 1")
    ax.axvline(med_x, color="gray", linestyle=":", label="Median Lab 2")
    if doe:
        ax.scatter(
            data.x, data.y, marker="x", color="k", zorder=5, label="DOE points"
        )
    ax.set_xlabel("Lab 2 (X)")
    ax.set_ylabel("Lab 1 (Y)")
    ax.set_title("Youden Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def star_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a star (radar) plot of multivariate data.

    Each observation is drawn as a polygon on a polar axis, with one
    spoke per variable. Values are normalized 0-1 per variable.

    Args:
        data: EDAData container. Requires factors for additional
            variables beyond y.
        fig: Matplotlib figure. If None, creates new polar figure.
        ax: Matplotlib axes (polar). If None, creates new polar axes.

    Returns:
        The figure and axes containing the plot.

    Raises:
        ValueError: If factors is None.
    """
    if not data.factors:
        msg = "star_plot requires factors"
        raise ValueError(msg)
    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    elif fig is not None and ax is None:
        ax = fig.add_subplot(1, 1, 1, projection="polar")
    elif fig is None:
        fig = ax.get_figure()
    var_names = ["y", *list(data.factors.keys())]
    factor_cols = [data.factors[k] for k in data.factors]
    all_vals = np.column_stack([data.y, *factor_cols])
    col_min = all_vals.min(axis=0)
    col_max = all_vals.max(axis=0)
    span = col_max - col_min
    span[span == 0] = 1.0
    normed = (all_vals - col_min) / span
    n_vars = len(var_names)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    closed_angles = np.append(angles, angles[0])
    for row in normed:
        closed_vals = np.append(row, row[0])
        ax.plot(closed_angles, closed_vals)
        ax.fill(closed_angles, closed_vals, alpha=0.1)
    ax.set_xticks(angles)
    ax.set_xticklabels(var_names)
    ax.set_ylim(0, 1)
    ax.set_title("Star Plot")
    fig.tight_layout()
    return fig, ax
