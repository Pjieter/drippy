"""Plotting functions for multi-factor/DOE models."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from drippy.data import EDAData


def doe_scatter_plot(
    data: EDAData,
    fig: Figure | None = None,
    axes: np.ndarray | None = None,
) -> tuple[Figure, np.ndarray]:
    """Creates scatter plots of y vs each factor.

    One subplot per factor in data.factors.

    Args:
        data: EDAData container. Requires factors.
        fig: Matplotlib figure. If None, creates new figure.
        axes: 1-D array of Axes, one per factor. If None,
            creates new axes.

    Returns:
        (fig, axes) where axes is a 1-D array of Axes.
    """
    if data.factors is None:
        msg = "doe_scatter_plot requires factors"
        raise ValueError(msg)
    n = len(data.factors)
    if fig is None and axes is None:
        fig, axes = plt.subplots(1, n)
        axes = np.atleast_1d(axes)
    elif axes is None:
        axes = np.atleast_1d(fig.subplots(1, n))
    elif fig is None:
        axes = np.atleast_1d(axes)
        fig = axes.flat[0].get_figure()
    axes = np.asarray(axes)
    if axes.shape != (n,):
        msg = f"axes must have shape ({n},)"
        raise ValueError(msg)
    for ax, (name, factor) in zip(axes, data.factors.items(), strict=False):
        ax.scatter(factor, data.y)
        ax.set_xlabel(name)
        ax.set_ylabel("Y")
        ax.set_title(f"Scatter Plot: {name}")
    fig.tight_layout()
    return fig, axes


def doe_mean_plot(
    data: EDAData,
    fig: Figure | None = None,
    axes: np.ndarray | None = None,
) -> tuple[Figure, np.ndarray]:
    """Creates mean plots of y grouped by each factor's levels.

    Shows group means connected by a line and a horizontal grand-mean
    reference line for each factor.

    Args:
        data: EDAData container. Requires factors.
        fig: Matplotlib figure. If None, creates new figure.
        axes: 1-D array of Axes, one per factor. If None,
            creates new axes.

    Returns:
        (fig, axes) where axes is a 1-D array of Axes.
    """
    if data.factors is None:
        msg = "doe_mean_plot requires factors"
        raise ValueError(msg)
    n = len(data.factors)
    if fig is None and axes is None:
        fig, axes = plt.subplots(1, n)
        axes = np.atleast_1d(axes)
    elif axes is None:
        axes = np.atleast_1d(fig.subplots(1, n))
    elif fig is None:
        axes = np.atleast_1d(axes)
        fig = axes.flat[0].get_figure()
    axes = np.asarray(axes)
    if axes.shape != (n,):
        msg = f"axes must have shape ({n},)"
        raise ValueError(msg)
    grand_mean = data.y.mean()
    for ax, (name, factor) in zip(axes, data.factors.items(), strict=False):
        levels = np.unique(factor)
        means = [data.y[factor == lvl].mean() for lvl in levels]
        ax.plot(levels, means, "o-")
        ax.axhline(
            grand_mean, color="r", linestyle="--", label="Grand mean"
        )
        ax.set_xlabel(name)
        ax.set_ylabel("Mean of Y")
        ax.set_title(f"Mean Plot: {name}")
        ax.legend()
    fig.tight_layout()
    return fig, axes


def doe_sd_plot(
    data: EDAData,
    fig: Figure | None = None,
    axes: np.ndarray | None = None,
) -> tuple[Figure, np.ndarray]:
    """Creates standard deviation plots of y by each factor's levels.

    Shows group standard deviations connected by a line and a horizontal
    overall-SD reference line for each factor.

    Args:
        data: EDAData container. Requires factors.
        fig: Matplotlib figure. If None, creates new figure.
        axes: 1-D array of Axes, one per factor. If None,
            creates new axes.

    Returns:
        (fig, axes) where axes is a 1-D array of Axes.
    """
    if data.factors is None:
        msg = "doe_sd_plot requires factors"
        raise ValueError(msg)
    n = len(data.factors)
    if fig is None and axes is None:
        fig, axes = plt.subplots(1, n)
        axes = np.atleast_1d(axes)
    elif axes is None:
        axes = np.atleast_1d(fig.subplots(1, n))
    elif fig is None:
        axes = np.atleast_1d(axes)
        fig = axes.flat[0].get_figure()
    axes = np.asarray(axes)
    if axes.shape != (n,):
        msg = f"axes must have shape ({n},)"
        raise ValueError(msg)
    overall_sd = data.y.std()
    for ax, (name, factor) in zip(axes, data.factors.items(), strict=False):
        levels = np.unique(factor)
        sds = [data.y[factor == lvl].std() for lvl in levels]
        ax.plot(levels, sds, "o-")
        ax.axhline(
            overall_sd, color="r", linestyle="--", label="Overall SD"
        )
        ax.set_xlabel(name)
        ax.set_ylabel("Standard Deviation of Y")
        ax.set_title(f"SD Plot: {name}")
        ax.legend()
    fig.tight_layout()
    return fig, axes


def contour_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    doe: bool = False,
) -> tuple[Figure, Axes]:
    """Creates a contour plot of y over the 2D factor space.

    Uses tricontourf for robustness with irregular/DOE grids.

    Args:
        data: EDAData container. Requires exactly 2 factors.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        doe: If True, overlays design point markers.

    Returns:
        The figure and axes containing the plot.
    """
    if data.factors is None:
        msg = "contour_plot requires factors"
        raise ValueError(msg)
    if len(data.factors) != 2:  # noqa: PLR2004
        n_factors = len(data.factors)
        msg = (
            "contour_plot requires exactly 2 factors, "
            f"got {n_factors}"
        )
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    names = list(data.factors.keys())
    f1 = data.factors[names[0]]
    f2 = data.factors[names[1]]
    cf = ax.tricontourf(f1, f2, data.y)
    fig.colorbar(cf, ax=ax, label="Y")
    if doe:
        ax.scatter(f1, f2, color="k", marker="x", label="Design points")
        ax.legend()
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Contour Plot")
    fig.tight_layout()
    return fig, ax
