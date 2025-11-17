"""Plotting functions for time series models."""

from collections.abc import Iterable
import numpy as np
import scipy as sp
from astropy.timeseries import LombScargle
from lmfit.models import LinearModel
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from drippy.utilities import get_figure_and_axes


class TimeSeriesPlotter:
    """A class for plotting time series models."""

    def __init__(self, y: Iterable[float], t: Iterable[float]) -> None:
        """Initialize TimeSeriesPlotter with data.

        Args:
            y (Iterable[float]):
                Sequentially ordered data points
            t (Iterable[float]):
                Sequentially ordered "time" points. Can be any
                continuously changing variable.
        """
        self.y = np.asarray(y)
        self.t = np.asarray(t)
        if self.y.size == 0 or self.t.size == 0:
            error = "y and t cannot be empty"
            raise ValueError(error)
        if self.y.ndim != 1 or self.t.ndim != 1:
            error = "y and t must be 1-dimensional arrays"
            raise ValueError(error)
        if len(self.t) != len(self.y):
            error = (
                f"t and y must have the same length. "
                f"Got len(t)={len(self.t)}, len(y)={len(self.y)}."
            )
            raise ValueError(error)

    def sequence_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Creates a sequence plot showing data over time.

        Args:
            fig (Optional[Figure], optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Optional[Axes], optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.

        Returns:
            tuple[Figure, Axes]:
                The figure and axes containing the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)
        ax.plot(self.t, self.y, label="Data")
        ax.legend()
        ax.set_xlabel("Time-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Sequence Plot")
        fig.tight_layout()
        return fig, ax

    def spectral_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        alarm_levels: bool = True,
    ) -> tuple[Figure, Axes]:
        """Creates a Lomb-Scargle periodogram.

        Args:
            fig (Figure, optional): Figure for plotting. If None,
                creates new figure. Defaults to None.
            ax (Axes, optional): Axes for plotting. If None,
                creates new axes. Defaults to None.
            alarm_levels (bool, optional): Whether to show false
                alarm levels. Defaults to True.

        Returns:
            tuple[Figure, Axes]: Figure and Axes with the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)

        ls = LombScargle(self.t, self.y)
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
        ax.set_xlabel("Frequency (cycles per unit time)")
        ax.set_ylabel("Spectral Power Density")
        ax.set_title("Lomb-Scargle Periodogram")
        fig.tight_layout()
        return fig, ax

    def auto_correlation_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Creates autocorrelation plot with confidence intervals.

        Args:
            fig (Optional[Figure], optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Optional[Axes], optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.

        Returns:
            tuple[Figure, Axes]: Figure and axes with the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)
        n = len(self.y)

        ax.acorr(self.y, usevlines=True, maxlags=n - 1)
        conf_interval = [0.99, 0.95, 0.8]
        for i, ci in enumerate(conf_interval):
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

    def complex_demodulation_phase_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Creates plot of instantaneous phase via Hilbert.

        Args:
            fig (Optional[Figure], optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Optional[Axes], optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.

        Returns:
            tuple[Figure, Axes]: Figure and axes with the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)

        analytic_signal = sp.signal.hilbert(self.y)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        model = LinearModel()
        params = model.make_params(intercept=0, slope=0)
        result = model.fit(instantaneous_phase, params, x=self.t)

        ax.plot(self.t, instantaneous_phase, label="Instantaneous Phase")
        ax.plot(
            self.t,
            result.best_fit,
            color="red",
            label=f"Linear Fit with R$^2$={result.rsquared:.3g}\n"
            rf"$\phi_0$={result.params['intercept'].value:.3g}, "
            rf"$\omega$={result.params['slope'].value:.3g}",
        )
        ax.set_xlabel("Time-axis")
        ax.set_ylabel("Phase (radians)")
        ax.set_title("Complex Demodulation Phase Plot")
        ax.legend()
        fig.tight_layout()
        return fig, ax
