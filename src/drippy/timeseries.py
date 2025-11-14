"""Contains the plotting functions for time series models."""

from typing import Iterable
from lmfit.model import ModelResult
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.timeseries import LombScargle
from lmfit.models import LinearModel


class TimeSeriesPlotter:
    """A class for plotting time series models."""

    def __init__(
        self, y: Iterable[float], x: Iterable[float], model_result: ModelResult
    ) -> None:
        """_summary_

        Args:
            y (Iterable[float]): Sequenciatially ordered data points
            x (Iterable[float]): Sequentially ordered "time" points
            model_result (ModelResult): The result of the model fitting
        """
        self.y = y
        self.x = x
        self.model_result = model_result

    def auto_plot(self) -> None:
        """Plots the given data."""
        # Implementation of the plotting logic goes here
        pass

    def sequence_plot(self, fig=None, ax=None) -> tuple[Figure, Axes]:
        """Creates a sequence plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.x, self.y, label="Data")
        ax.plot(self.x, self.model_result.best_fit, color="red", label="Best Fit")
        ax.legend()
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Sequence Plot with Best Fit")
        fig.show()
        return fig, ax

    def spectral_plot(
        self, fig: Figure = None, ax: Axes = None, alarm_levels: bool = True
    ) -> tuple[Figure, Axes]:
        """Creates a spectral plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        residuals = self.model_result.residual

        ls = LombScargle(self.x, residuals)
        frequency, power = ls.autopower(normalization="psd")
        ax.plot(frequency, power, label="Data")

        if alarm_levels:
            false_alarm_max_peak = ls.false_alarm_probability(power.max())
            false_alarm_levels = ls.false_alarm_level([0.1, 0.05, 0.01])
            ax.plot(
                frequency[np.argmax(power)],
                power.max(),
                "rx",
                label=f"False alarm level max peak: {false_alarm_max_peak*100:.3g}%",
            )
            for i, fal in enumerate(false_alarm_levels):
                ax.axhline(
                    fal,
                    color=f"C{i+1}",
                    linestyle="--",
                    label=f"False Alarm Level {['10%', '5%', '1%'][i]}",
                )
        ax.legend()
        ax.set_xlabel("Frequency (cycles per unit time)")
        ax.set_ylabel("Spectral Power Density")
        return fig, ax

    def auto_correlation_plot(self, fig=None, ax=None) -> tuple[Figure, Axes]:
        """Creates an autocorrelation plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        residuals = self.model_result.residual

        _, c, _, _ = ax.acorr(residuals, usevlines=True, maxlags=len(residuals) - 1)
        ax.set_xlim(0)
        # ax.set_ylim(c.min(), np.partition(c.flatten(), -2)[-2])
        conf_interval = [0.99, 0.95, 0.8]
        for i, ci in enumerate(conf_interval):
            conf_level = sp.stats.norm.cdf(1 - ci / 2) / np.sqrt(len(residuals))
            ax.axhline(
                conf_level,
                color=f"C{i+1}",
                linestyle="--",
                label=f"{ci*100:.0f}% confidence level",
            )
            ax.axhline(-conf_level, color=f"C{i+1}", linestyle="--")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def complex_demodulation_phase_plot(
        self, fig: Figure = None, ax: Axes = None
    ) -> tuple[Figure, Axes]:
        """Creates a complex demodulation phase plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        analytic_signal = sp.signal.hilbert(self.y)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        model = LinearModel()
        params = model.make_params(intercept=0, slope=0)
        result = model.fit(instantaneous_phase, params, x=self.x)

        ax.plot(self.x, instantaneous_phase, label="Instantaneous Phase")
        ax.plot(
            self.x,
            result.best_fit,
            color="red",
            label=f"Linear Fit with R$^2$={result.rsquared:.3g}\n"
            f"$\phi_0$={result.params['intercept'].value:.3g}, $\omega$={result.params['slope'].value:.3g}",
        )
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Phase (radians)")
        ax.set_title("Complex Demodulation Phase Plot")
        ax.legend()
        fig.tight_layout()
        return fig, ax
