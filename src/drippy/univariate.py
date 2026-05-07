"""Plotting functions for univariate models (y = c + e)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from lmfit.models import LinearModel
from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from collections.abc import Callable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from drippy.data import EDAData


def run_sequence_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a run sequence plot of y vs index or continuous index t.

    Args:
        data: EDAData container. Uses t as x-axis if present, else index.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    fig, ax = get_figure_and_axes(fig, ax)
    x_vals = data.t if data.t is not None else np.arange(len(data.y))
    ax.plot(x_vals, data.y, label="Data")
    ax.legend()
    ax.set_xlabel("Index" if data.t is None else "t")
    ax.set_ylabel("Y")
    ax.set_title("Run Sequence Plot")
    fig.tight_layout()
    return fig, ax


def lag_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    lag: int = 1,
) -> tuple[Figure, Axes]:
    """Creates a lag plot of the data.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        lag: Number of lags. Must be positive and less than len(y).

    Returns:
        The figure and axes containing the plot.
    """
    lag = int(lag)
    if lag <= 0:
        msg = "Lag must be a positive integer"
        raise ValueError(msg)
    if lag >= len(data.y):
        msg = "Lag must be less than the length of the data"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    y_original = data.y[lag:]
    y_lagged = data.y[:-lag]
    colors = np.arange(len(y_lagged))
    scatter = ax.scatter(
        y_lagged, y_original, c=colors, cmap="viridis", label="Lag Plot"
    )
    fig.colorbar(scatter, ax=ax, label="Index")
    ax.set_xlabel(rf"Y$_{{i-{lag}}}$")
    ax.set_ylabel("Y$_{i}$")
    ax.set_title(f"Lag Plot (lag={lag})")
    fig.tight_layout()
    return fig, ax


def histogram(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    bins: int | str = "auto",
) -> tuple[Figure, Axes]:
    """Creates a histogram of the data.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        bins: Number of bins or bin strategy.

    Returns:
        The figure and axes containing the plot.
    """
    if isinstance(bins, int) and bins <= 0:
        msg = f"Number of bins must be positive, got {bins}"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    ax.hist(data.y, bins=bins)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Y values")
    fig.tight_layout()
    return fig, ax


def normal_probability_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    *,
    return_rsquared: bool = False,
) -> tuple[Figure, Axes] | tuple[Figure, Axes, float]:
    """Creates a normal probability plot.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        return_rsquared: If True, returns R-squared as third element.

    Returns:
        (fig, ax) or (fig, ax, r_squared) if return_rsquared is True.
    """
    fig, ax = get_figure_and_axes(fig, ax)
    _, (_, _, r_value) = sp.stats.probplot(
        data.y, dist="norm", plot=ax, rvalue=True
    )
    ax.set_ylabel("Ordered Values")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_title("Normal Probability Plot")
    fig.tight_layout()
    if return_rsquared:
        return fig, ax, r_value**2
    return fig, ax


def four_plot(
    data: EDAData,
    fig: Figure | None = None,
    axes: np.ndarray | None = None,
) -> tuple[Figure, np.ndarray]:
    """Creates a 4-plot (run sequence, lag, histogram, normal probability).

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        axes: ndarray of Axes with shape (2, 2). If None, creates new axes.

    Returns:
        (fig, axes_flat) where axes_flat has shape (4,).
    """
    if fig is None and axes is None:
        fig, axes = plt.subplots(2, 2)
    elif axes is None:
        axes = fig.subplots(2, 2)
    elif fig is None:
        fig = axes.flat[0].get_figure()
    if axes.shape != (2, 2):
        msg = "Axes must be an iterable of (2, 2) Axes objects."
        raise ValueError(msg)
    axes = axes.flatten()
    run_sequence_plot(data, fig, axes[0])
    lag_plot(data, fig, axes[1])
    histogram(data, fig, axes[2])
    normal_probability_plot(data, fig, axes[3])
    return fig, axes


def ppcc_plot(  # noqa: PLR0913
    data: EDAData,
    fig: Figure | None = None,
    ax: np.ndarray | None = None,
    rough_range: tuple[float, float] = (-2, 2),
    n_rough: int = 50,
    n_fine: int = 100,
) -> tuple[Figure, np.ndarray]:
    """Creates a PPCC plot (rough + fine) for distribution shape estimation.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: ndarray of Axes with shape (2,). If None, creates new axes.
        rough_range: (min, max) range for rough search.
        n_rough: Points in rough plot.
        n_fine: Points in fine plot.

    Returns:
        (fig, axes) where axes has shape (2,).
    """
    if len(rough_range) != 2:  # noqa: PLR2004
        msg = "Rough range must contain exactly 2 elements"
        raise ValueError(msg)
    if rough_range[0] >= rough_range[1]:
        msg = "Rough range must be (min, max) with min < max"
        raise ValueError(msg)
    if n_rough <= 0:
        msg = "Number of points must be positive"
        raise ValueError(msg)
    if n_fine <= 0:
        msg = "Number of points must be positive"
        raise ValueError(msg)
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 2)
    elif ax is None:
        ax = fig.subplots(1, 2)
    elif fig is None:
        fig = ax.flat[0].get_figure()
    if ax.shape != (2,):
        msg = "Axes must be an iterable of 2 Axes objects."
        raise ValueError(msg)
    rough_shape_values, rough_ppcc = sp.stats.ppcc_plot(
        data.y,
        rough_range[0],
        rough_range[1],
        N=n_rough,
        plot=ax[0],
    )
    rough_max_index = np.argmax(rough_ppcc)
    fine_shape_values, fine_ppcc = sp.stats.ppcc_plot(
        data.y,
        rough_shape_values[rough_max_index] - 0.5,
        rough_shape_values[rough_max_index] + 0.5,
        N=n_fine,
        plot=ax[1],
    )
    fine_max_index = np.argmax(fine_ppcc)
    max_shape = rough_shape_values[rough_max_index]
    ax[0].vlines(
        max_shape,
        0,
        rough_ppcc[rough_max_index],
        color="r",
        label=f"Max PPCC at shape={max_shape:.3g}",
    )
    ax[0].legend()
    max_shape_fine = fine_shape_values[fine_max_index]
    ax[1].axvline(
        max_shape_fine,
        color="r",
        label=f"Max PPCC at shape={max_shape_fine:.3g}",
    )
    ax[1].legend()
    ax[0].set_title("Rough PPCC Plot")
    ax[1].set_title("Fine PPCC Plot")
    fig.tight_layout()
    return fig, ax


def weibull_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a Weibull probability plot.

    Args:
        data: EDAData container. Requires y > 0.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if not np.all(data.y > 0):
        msg = "Weibull plot requires all y values to be positive."
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    n = len(data.y)
    p = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    weibull_probabilities = np.log(-np.log(1 - p))
    ordered_data = np.log(np.sort(data.y))
    model = LinearModel()
    params = model.make_params(intercept=0, slope=1)
    result = model.fit(weibull_probabilities, params, x=ordered_data)
    shape = result.params["slope"].value
    intercept = result.params["intercept"].value
    scale = np.exp(-intercept / shape)
    x_fit = np.linspace(min(ordered_data), max(ordered_data), 100)
    y_fit = intercept + shape * x_fit
    ax.plot(ordered_data, weibull_probabilities, ".", label="Data")
    ax.plot(
        x_fit,
        y_fit,
        "-",
        label=f"Fit with R$^2$={result.rsquared:.3g}",
    )
    ax.set_xlabel("Ordered Values")
    ax.set_ylabel("Theoretical Quantiles (Weibull)")
    ax.set_title("Weibull Probability Plot")
    min_percentage = np.floor(
        np.log10(1 - np.exp(np.exp(min(weibull_probabilities)) * -1)),
    )
    percentages = np.concatenate(
        (
            np.logspace(min_percentage - 1, -1, num=int(-min_percentage) + 1),
            [0.5, 0.9, 0.99],
        ),
    )
    ax.set_yticks(
        np.log(-np.log(1 - percentages)),
        labels=[f"{p * 100:.3g}" for p in percentages],
    )
    ax.axhline(np.log(-np.log(1 - 0.632)), color="black", linestyle="--")
    ax.axvline(np.log(scale), color="black", linestyle="--")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def probability_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    distribution: str = "norm",
) -> tuple[Figure, Axes]:
    """Creates a probability plot for a specified distribution.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        distribution: scipy.stats distribution name.

    Returns:
        The figure and axes containing the plot.
    """
    fig, ax = get_figure_and_axes(fig, ax)
    sp.stats.probplot(data.y, dist=distribution, plot=ax)
    ax.set_ylabel("Ordered Values")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_title(f"{distribution.capitalize()} Probability Plot")
    fig.tight_layout()
    return fig, ax


def box_cox_normality_plot(
    data: EDAData,
    fig: Figure | None = None,
    axes: np.ndarray | None = None,
) -> tuple[Figure, np.ndarray]:
    """Creates a Box-Cox normality plot (2x2 grid).

    Shows: original histogram, Box-Cox normality curve, transformed
    histogram, normal probability plot of transformed data.

    Args:
        data: EDAData container. Requires y > 0.
        fig: Matplotlib figure. If None, creates new figure.
        axes: ndarray of Axes with shape (2, 2). If None, creates new.

    Returns:
        (fig, axes) where axes has shape (2, 2).
    """
    if not np.all(data.y > 0):
        msg = "Box-Cox transformation requires all y values to be positive."
        raise ValueError(msg)
    if fig is None and axes is None:
        fig, axes = plt.subplots(2, 2)
    elif axes is None:
        axes = fig.subplots(2, 2)
    elif fig is None:
        fig = axes.flat[0].get_figure()
    if axes.shape != (2, 2):
        msg = "Axes must be an iterable of (2, 2) Axes objects."
        raise ValueError(msg)
    sp.stats.boxcox_normplot(data.y, -2, 2, plot=axes[0, 1], N=200)
    transformed_y, maxlog = sp.stats.boxcox(data.y)
    axes[0, 0].hist(data.y, bins="auto")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Histogram of Original (positive shifted) Y values")
    axes[0, 1].set_title("Box-Cox Normality Plot")
    axes[0, 1].axvline(
        maxlog, color="r", linestyle="--", label=f"Max log={maxlog:.3g}"
    )
    axes[0, 1].legend()
    axes[1, 0].hist(transformed_y, bins="auto")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Histogram of transformed Y values")
    sp.stats.probplot(transformed_y, dist="norm", plot=axes[1, 1], rvalue=True)
    axes[1, 1].set_title("Normal Probability Plot of Transformed Data")
    fig.tight_layout()
    return fig, axes


def bootstrap_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
) -> tuple[Figure, Axes]:
    """Creates a bootstrap distribution plot.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        statistic: Callable to bootstrap. Must be callable.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        The figure and axes containing the plot.
    """
    if not callable(statistic):
        msg = "Statistic must be callable"
        raise TypeError(msg)
    if n_bootstrap <= 0:
        msg = "Number of bootstrap samples must be positive"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    result = sp.stats.bootstrap((data.y,), statistic, n_resamples=n_bootstrap)
    variability_label = (
        f"Variability of {statistic.__name__}: {result.standard_error:.3g}"
    )
    ax.hist(
        result.bootstrap_distribution,
        bins="auto",
        density=True,
        label=variability_label,
    )
    ax.set_xlabel(f"Bootstrap {statistic.__name__} values")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap Distribution")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def box_cox_linearity_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    lmbda_range: tuple[float, float] = (-2, 2),
    n: int = 100,
) -> tuple[Figure, Axes]:
    """Creates a Box-Cox linearity plot.

    Plots |corr(Y, X^λ)| across a range of λ values to find the
    power transformation of X that maximises linearity with Y.

    Args:
        data: EDAData container. Requires x > 0.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        lmbda_range: (min, max) range of λ to evaluate.
        n: Number of λ values to evaluate.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "box_cox_linearity_plot requires x"
        raise ValueError(msg)
    if not np.all(data.x > 0):
        msg = "box_cox_linearity_plot requires all x values to be positive"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    lambdas = np.linspace(lmbda_range[0], lmbda_range[1], n)
    correlations = np.array(
        [
            abs(
                np.corrcoef(
                    np.log(data.x)
                    if lmbda == 0
                    else (data.x**lmbda - 1) / lmbda,
                    data.y,
                )[0, 1]
            )
            for lmbda in lambdas
        ]
    )
    optimal_lambda = lambdas[np.argmax(correlations)]
    ax.plot(lambdas, correlations)
    ax.axvline(
        optimal_lambda,
        color="r",
        linestyle="--",
        label=f"Optimal λ={optimal_lambda:.3g}",
    )
    ax.set_xlabel("λ (power transformation)")
    ax.set_ylabel("|Correlation(Y, X^λ)|")
    ax.set_title("Box-Cox Linearity Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax
