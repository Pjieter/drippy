"""Plotting functions for regression models (y = f(x) + e, x continuous)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from drippy.onefactor import scatter_plot  # shared — not re-implemented
from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from drippy.data import EDAData

__all__ = [
    "linear_correlation_plot",
    "linear_intercept_plot",
    "linear_residual_sd_plot",
    "linear_slope_plot",
    "scatter_plot",
    "six_plot",
]


def _rolling_linregress(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling OLS for each window of data.

    Returns:
        (midpoints, slopes, intercepts, residual_sds) one value per window.
    """
    n = len(y)
    n_windows = n - window + 1
    midpoints = np.empty(n_windows)
    slopes = np.empty(n_windows)
    intercepts = np.empty(n_windows)
    residual_sds = np.empty(n_windows)
    for i in range(n_windows):
        x_win = x[i : i + window].astype(float)
        y_win = y[i : i + window]
        result = scipy.stats.linregress(x_win, y_win)
        midpoints[i] = i + window // 2
        slopes[i] = result.slope
        intercepts[i] = result.intercept
        res = y_win - (result.slope * x_win + result.intercept)
        residual_sds[i] = res.std()
    return midpoints, slopes, intercepts, residual_sds


def six_plot(  # noqa: PLR0915
    data: EDAData,
    fig: Figure | None = None,
    axes: np.ndarray | None = None,
) -> tuple[Figure, np.ndarray]:
    """Creates a 2x3 composite regression diagnostic plot.

    The six panels are: scatter with regression line, residuals vs x, lag
    plot of residuals, histogram of residuals, normal probability plot of
    residuals, and run sequence of residuals.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        axes: 2x3 array of Axes. If None, creates new axes.

    Returns:
        (fig, axes) where axes has shape (2, 3).
    """
    if data.x is None:
        msg = "six_plot requires x"
        raise ValueError(msg)
    if fig is None and axes is None:
        fig, axes = plt.subplots(2, 3)
    elif axes is None:
        axes = fig.subplots(2, 3)
    elif fig is None:
        fig = axes.flat[0].get_figure()
    if axes.shape != (2, 3):
        msg = "axes must have shape (2, 3)"
        raise ValueError(msg)

    x = data.x.astype(float)
    y = data.y
    result = scipy.stats.linregress(x, y)
    slope, intercept = result.slope, result.intercept
    res = y - (slope * x + intercept)

    # [0,0] Scatter plot with regression line
    ax = axes[0, 0]
    ax.scatter(x, y, s=10)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, color="r")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter Plot")

    # [0,1] Residuals vs x
    ax = axes[0, 1]
    ax.scatter(x, res, s=10)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("X")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")

    # [0,2] Lag plot of residuals (lag 1)
    ax = axes[0, 2]
    ax.scatter(res[:-1], res[1:], s=10)
    ax.set_xlabel("Residual[i]")
    ax.set_ylabel("Residual[i+1]")
    ax.set_title("Lag Plot")

    # [1,0] Histogram of residuals
    ax = axes[1, 0]
    ax.hist(res, bins="auto")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram")

    # [1,1] Normal probability plot of residuals
    ax = axes[1, 1]
    (osm, osr), _ = scipy.stats.probplot(res, dist="norm")
    ax.scatter(osm, osr, s=10)
    ax.plot(
        [osm[0], osm[-1]],
        [osm[0], osm[-1]],
        color="r",
        linestyle="--",
    )
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Ordered Residuals")
    ax.set_title("Normal Probability Plot")

    # [1,2] Run sequence of residuals
    ax = axes[1, 2]
    ax.plot(res)
    ax.axhline(res.mean(), color="r", linestyle="--")
    ax.set_xlabel("Index")
    ax.set_ylabel("Residuals")
    ax.set_title("Run Sequence")

    fig.tight_layout()
    return fig, axes


def linear_correlation_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    window: int = 10,
) -> tuple[Figure, Axes]:
    """Plots Pearson correlation coefficient for rolling windows of data.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        window: Number of observations per rolling window.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "linear_correlation_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    x = data.x.astype(float)
    y = data.y
    n = len(y)
    n_windows = n - window + 1
    midpoints = np.empty(n_windows)
    correlations = np.empty(n_windows)
    for i in range(n_windows):
        r, _ = scipy.stats.pearsonr(x[i : i + window], y[i : i + window])
        midpoints[i] = i + window // 2
        correlations[i] = r
    ax.plot(midpoints, correlations, "o-", markersize=4)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Subset")
    ax.set_ylabel("Correlation r")
    ax.set_title("Linear Correlation Plot")
    fig.tight_layout()
    return fig, ax


def linear_intercept_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    window: int = 10,
) -> tuple[Figure, Axes]:
    """Plots OLS regression intercept for rolling windows of data.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        window: Number of observations per rolling window.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "linear_intercept_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    midpoints, _, intercepts, _ = _rolling_linregress(
        data.x.astype(float), data.y, window
    )
    ax.plot(midpoints, intercepts, "o-", markersize=4)
    ax.axhline(intercepts.mean(), color="r", linestyle="--")
    ax.set_xlabel("Subset")
    ax.set_ylabel("Intercept")
    ax.set_title("Linear Intercept Plot")
    fig.tight_layout()
    return fig, ax


def linear_slope_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    window: int = 10,
) -> tuple[Figure, Axes]:
    """Plots OLS regression slope for rolling windows of data.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        window: Number of observations per rolling window.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "linear_slope_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    midpoints, slopes, _, _ = _rolling_linregress(
        data.x.astype(float), data.y, window
    )
    ax.plot(midpoints, slopes, "o-", markersize=4)
    ax.axhline(slopes.mean(), color="r", linestyle="--")
    ax.set_xlabel("Subset")
    ax.set_ylabel("Slope")
    ax.set_title("Linear Slope Plot")
    fig.tight_layout()
    return fig, ax


def linear_residual_sd_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    window: int = 10,
) -> tuple[Figure, Axes]:
    """Plots residual standard deviation for rolling windows of data.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        window: Number of observations per rolling window.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "linear_residual_sd_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    midpoints, _, _, residual_sds = _rolling_linregress(
        data.x.astype(float), data.y, window
    )
    ax.plot(midpoints, residual_sds, "o-", markersize=4)
    ax.axhline(residual_sds.mean(), color="r", linestyle="--")
    ax.set_xlabel("Subset")
    ax.set_ylabel("Residual SD")
    ax.set_title("Linear Residual SD Plot")
    fig.tight_layout()
    return fig, ax
