"""Plotting functions for time series models (y = f(t) + e)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy as sp
from astropy.timeseries import LombScargle
from lmfit.models import LinearModel
from drippy.univariate import run_sequence_plot  # shared function
from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from drippy.data import EDAData

__all__ = [
    "autocorrelation_plot",
    "complex_demodulation_amplitude_plot",
    "complex_demodulation_phase_plot",
    "run_sequence_plot",
    "spectral_plot",
]


def spectral_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    alarm_levels: bool = True,
) -> tuple[Figure, Axes]:
    """Creates a Lomb-Scargle periodogram.

    Args:
        data: EDAData container. Requires t.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        alarm_levels: Whether to show false alarm probability levels.

    Returns:
        The figure and axes containing the plot.
    """
    if data.t is None:
        msg = "spectral_plot requires t (continuous index)"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    ls = LombScargle(data.t, data.y)
    frequency, power = ls.autopower(normalization="psd")
    ax.plot(frequency, power, label="Data")
    if alarm_levels:
        false_alarm_max_peak = ls.false_alarm_probability(power.max())
        false_alarm_levels = ls.false_alarm_level([0.1, 0.05, 0.01])
        ax.plot(
            frequency[np.argmax(power)],
            power.max(),
            "rx",
            label=(
                f"False alarm level max peak: "
                f"{false_alarm_max_peak * 100:.3g}%"
            ),
        )
        for i, fal in enumerate(false_alarm_levels):
            ax.axhline(
                fal,
                color=f"C{i + 1}",
                linestyle="--",
                label=f"False Alarm Level {['10%', '5%', '1%'][i]}",
            )
    ax.legend()
    ax.set_xlabel("Frequency (cycles per unit t)")
    ax.set_ylabel("Spectral Power Density")
    ax.set_title("Lomb-Scargle Periodogram")
    fig.tight_layout()
    return fig, ax


def autocorrelation_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates autocorrelation plot with confidence intervals.

    Includes 99%, 95%, and 80% confidence intervals.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    fig, ax = get_figure_and_axes(fig, ax)
    n = len(data.y)
    ax.acorr(data.y, usevlines=True, maxlags=n - 1)
    for i, ci in enumerate([0.99, 0.95, 0.8]):
        conf_level = sp.stats.norm.ppf((1 + ci) / 2) / np.sqrt(n)
        ax.axhline(
            conf_level,
            color=f"C{i + 1}",
            linestyle="--",
            label=f"{ci * 100:.0f}% confidence level",
        )
        ax.axhline(-conf_level, color=f"C{i + 1}", linestyle="--")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def complex_demodulation_amplitude_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates instantaneous amplitude plot via Hilbert transform.

    Args:
        data: EDAData container. Requires t.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.t is None:
        msg = (
            "complex_demodulation_amplitude_plot requires t (continuous index)"
        )
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    analytic_signal = sp.signal.hilbert(data.y)
    instantaneous_amplitude = np.abs(analytic_signal)
    ax.plot(data.t, instantaneous_amplitude, label="Instantaneous Amplitude")
    ax.set_xlabel("t")
    ax.set_ylabel("Amplitude")
    ax.set_title("Complex Demodulation Amplitude Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def complex_demodulation_phase_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates instantaneous phase plot via Hilbert transform with linear fit.

    Args:
        data: EDAData container. Requires t.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.t is None:
        msg = "complex_demodulation_phase_plot requires t (continuous index)"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    analytic_signal = sp.signal.hilbert(data.y)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    model = LinearModel()
    params = model.make_params(intercept=0, slope=0)
    result = model.fit(instantaneous_phase, params, x=data.t)
    ax.plot(data.t, instantaneous_phase, label="Instantaneous Phase")
    ax.plot(
        data.t,
        result.best_fit,
        color="red",
        label=(
            f"Linear Fit with R$^2$={result.rsquared:.3g}\n"
            rf"$\phi_0$={result.params['intercept'].value:.3g}, "
            rf"$\omega$={result.params['slope'].value:.3g}"
        ),
    )
    ax.set_xlabel("t")
    ax.set_ylabel("Phase (radians)")
    ax.set_title("Complex Demodulation Phase Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax
