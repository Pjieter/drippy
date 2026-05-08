"""Plotting functions for 1-factor models (y = f(x) + e, x categorical)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from drippy.data import EDAData

_FACTOR_LEVEL = "Factor Level"


def scatter_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a scatter plot of y vs x.

    Also used in regression context (see drippy.regression).

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "scatter_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    ax.scatter(data.x, data.y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter Plot")
    fig.tight_layout()
    return fig, ax


def box_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a box plot of y grouped by factor levels in x.

    Args:
        data: EDAData container. Requires x (categorical).
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "box_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    levels = np.unique(data.x)
    groups = [data.y[data.x == level] for level in levels]
    ax.boxplot(groups, tick_labels=levels)
    ax.set_xlabel(_FACTOR_LEVEL)
    ax.set_ylabel("Y")
    ax.set_title("Box Plot")
    fig.tight_layout()
    return fig, ax


def bihistogram(
    data: EDAData,
    fig: Figure | None = None,
    axes: np.ndarray | None = None,
    bins: int | str = "auto",
) -> tuple[Figure, np.ndarray]:
    """Creates side-by-side histograms for exactly 2 factor levels.

    Args:
        data: EDAData container. Requires x with exactly 2 unique levels.
        fig: Matplotlib figure. If None, creates new figure.
        axes: Array of 2 Axes. If None, creates new axes.
        bins: Number of bins or bin strategy.

    Returns:
        (fig, axes) where axes is a 1-D array of 2 Axes.
    """
    if data.x is None:
        msg = "bihistogram requires x"
        raise ValueError(msg)
    levels = np.unique(data.x)
    if len(levels) != 2:  # noqa: PLR2004
        n_levels = len(levels)
        msg = f"bihistogram requires exactly 2 factor levels, got {n_levels}"
        raise ValueError(msg)
    if fig is None and axes is None:
        fig, axes = plt.subplots(1, 2)
    elif axes is None:
        axes = fig.subplots(1, 2)
    elif fig is None:
        fig = axes.flat[0].get_figure()
    if axes.shape != (2,):
        msg = "axes must have shape (2,)"
        raise ValueError(msg)
    group_a = data.y[data.x == levels[0]]
    group_b = data.y[data.x == levels[1]]
    axes[0].hist(group_a, bins=bins)
    axes[0].set_title(f"Histogram: {levels[0]}")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(group_b, bins=bins)
    axes[1].set_title(f"Histogram: {levels[1]}")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    fig.tight_layout()
    return fig, axes


def qq_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a quantile-quantile plot comparing 2 factor level distributions.

    Args:
        data: EDAData container. Requires x with exactly 2 unique levels.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "qq_plot requires x"
        raise ValueError(msg)
    levels = np.unique(data.x)
    if len(levels) != 2:  # noqa: PLR2004
        n_levels = len(levels)
        msg = f"qq_plot requires exactly 2 factor levels, got {n_levels}"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    group_a = data.y[data.x == levels[0]]
    group_b = data.y[data.x == levels[1]]
    n = min(len(group_a), len(group_b))
    quantiles = np.linspace(0, 1, n)
    qa = np.quantile(group_a, quantiles)
    qb = np.quantile(group_b, quantiles)
    ax.scatter(qa, qb, label="Quantiles")
    min_val = min(qa.min(), qb.min())
    max_val = max(qa.max(), qb.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
    ax.set_xlabel(f"Quantiles: {levels[0]}")
    ax.set_ylabel(f"Quantiles: {levels[1]}")
    ax.set_title("Quantile-Quantile Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def mean_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a mean plot of y grouped by factor levels in x.

    Shows group means connected by a line, with a horizontal grand mean.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "mean_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    levels = np.unique(data.x)
    means = [data.y[data.x == level].mean() for level in levels]
    ax.plot(levels, means, "o-")
    ax.axhline(data.y.mean(), color="r", linestyle="--", label="Grand mean")
    ax.set_xlabel(_FACTOR_LEVEL)
    ax.set_ylabel("Mean of Y")
    ax.set_title("Mean Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def sd_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a standard deviation plot of y grouped by factor levels in x.

    Shows group standard deviations connected by a line, with a horizontal
    overall standard deviation reference line.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "sd_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    levels = np.unique(data.x)
    sds = [data.y[data.x == level].std() for level in levels]
    ax.plot(levels, sds, "o-")
    ax.axhline(data.y.std(), color="r", linestyle="--", label="Overall SD")
    ax.set_xlabel(_FACTOR_LEVEL)
    ax.set_ylabel("Standard Deviation of Y")
    ax.set_title("Standard Deviation Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax
