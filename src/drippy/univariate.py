"""Contains the plotting functions for Univariate (y = constant + error) models."""

from typing import Iterable
from lmfit.models import LinearModel
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


class UnivariatePlotter:
    """A class for plotting Univariate (y = constant + error) models."""

    def __init__(self, y: Iterable[float], x: Iterable[float]) -> None:
        """_summary_

        Args:
            y (Iterable[float]): Sequenciatially ordered data points
            x (Iterable[float]): Sequentially ordered "time" points
        """
        self.y = y
        self.x = x

    def auto_plot(self) -> None:
        """Plots the given data."""
        # Implementation of the plotting logic goes here
        pass

    def sequence_plot(self, fig=None, ax=None) -> tuple[Figure, Axes]:
        """Creates a sequence plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.x, self.y, label="Data")
        ax.legend()
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Sequence Plot")
        return fig, ax

    def lag_plot(
        self, fig: Figure = None, ax: Axes = None, lag: int = 1
    ) -> tuple[Figure, Axes]:
        """Creates a lag plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        y_lagged = self.y[lag:]
        y_original = self.y[:-lag]

        # Set color of point equal to it's index to create a gradient effect
        colors = np.arange(len(y_lagged))
        scatter = ax.scatter(
            y_lagged, y_original, c=colors, cmap="viridis", label="Lag Plot"
        )
        fig.colorbar(scatter, ax=ax, label="Index")
        ax.set_xlabel(rf"Y$_{{i-{lag}}}$")
        ax.set_ylabel("Y$_{i}$")
        ax.set_title(f"Lag Plot (lag={lag})")
        return fig, ax

    def histogram_plot(
        self, fig: Figure = None, ax: Axes = None, bins: int | str = "auto"
    ) -> tuple[Figure, Axes]:
        """Creates a histogram plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.hist(self.y, bins=bins)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Y values")
        return fig, ax

    def normal_probability_plot(
        self, fig: Figure = None, ax: Axes = None, return_Rsquared: bool = False
    ) -> tuple[Figure, Axes]:
        """Creates a normal probability plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        _, (_, _, R) = sp.stats.probplot(self.y, dist="norm", plot=ax, rvalue=True)
        ax.set_ylabel("Ordered Values")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_title("Normal Probability Plot")
        if return_Rsquared:
            return fig, ax, R**2
        return fig, ax

    def four_plot(
        self, fig: Figure = None, axes: Iterable[Axes] = None
    ) -> tuple[Figure, Iterable[Axes]]:
        """Creates a 4-plot of the data."""
        if fig is None or axes is None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        if axes.shape != (2, 2):
            raise ValueError("Axes must be an iterable of (2, 2) Axes objects.")

        axes = axes.flatten()

        # Sequence plot
        self.sequence_plot(fig, axes[0])

        # Lag plot
        self.lag_plot(fig, axes[1])

        # Histogram plot
        self.histogram_plot(fig, axes[2])

        # Normal probability plot
        self.normal_probability_plot(fig, axes[3])

        fig.tight_layout()
        return fig, axes

    def ppcc_plot(
        self,
        fig: Figure = None,
        ax: Axes = None,
        distribution: str | sp.stats.rv_continuous = "tukeylambda",
    ) -> tuple[Figure, Axes]:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 2)

        if ax.shape != (2,):
            raise ValueError("Axes must be an iterable of 2 Axes objects.")

        rough_shape_values, rough_ppcc = sp.stats.ppcc_plot(self.y, -2, 2, plot=ax[0])
        max_ppcc_index = np.argmax(rough_ppcc)
        fine_shape_values, fine_ppcc = sp.stats.ppcc_plot(
            self.y,
            rough_shape_values[max_ppcc_index] - 0.5,
            rough_shape_values[max_ppcc_index] + 0.5,
            N=100,
            plot=ax[1],
        )
        max_ppcc_index = np.argmax(fine_ppcc)
        ax[0].vlines(
            rough_shape_values[max_ppcc_index],
            0,
            rough_ppcc[max_ppcc_index],
            color="r",
            label=f"Max PPCC at shape={rough_shape_values[max_ppcc_index]:.3g}",
        )
        ax[0].legend()
        ax[1].axvline(
            fine_shape_values[max_ppcc_index],
            color="r",
            label=f"Max PPCC at shape={fine_shape_values[max_ppcc_index]:.3g}",
        )
        ax[1].legend()
        ax[0].set_title("Rough PPCC Plot")
        ax[1].set_title("Fine PPCC Plot")
        return fig, ax

    def weibull_plot(self, fig: Figure = None, ax: Axes = None) -> tuple[Figure, Axes]:
        """Creates a Weibull probability plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        assert np.all(
            np.array(self.y) > 0
        ), "Weibull plot requires all y values to be positive."

        # Create Weibull probability plot data
        n = len(self.y)
        p = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        weibull_probabilities = np.log(-np.log(1 - p))
        ordered_data = np.log(self.y)

        # Fit line
        model = LinearModel()
        params = model.make_params(intercept=0, slope=1)
        result = model.fit(weibull_probabilities, params, x=ordered_data)
        shape = result.params["slope"].value
        intercept = result.params["intercept"].value
        scale = np.exp(-intercept / shape)

        x_fit = np.linspace(min(ordered_data), max(ordered_data), 100)
        y_fit = intercept + shape * x_fit
        fig, ax = plt.subplots()
        ax.plot(ordered_data, weibull_probabilities, ".", label="Data")
        ax.plot(x_fit, y_fit, "-", label=f"Fit with R$^2$={result.rsquared:.3g}")
        ax.set_xlabel("Ordered Values")
        ax.set_ylabel("Theoretical Quantiles (Weibull)")
        ax.set_title("Weibull Probability Plot")
        min_percentage = np.floor(
            np.log10(1 - np.exp(np.exp(min(weibull_probabilities)) * -1))
        )
        percentages = np.concat(
            (
                np.logspace(min_percentage - 1, -1, num=int(-min_percentage) + 1),
                [0.5, 0.9, 0.99],
            )
        )
        ax.set_yticks(
            np.log(-np.log(1 - percentages)),
            labels=[f"{p*100:.3g}" for p in percentages],
        )
        ax.axhline(np.log(-np.log(1 - 0.632)), color="black", linestyle="--")
        ax.axvline(np.log(scale), color="black", linestyle="--")
        ax.legend()
        return fig, ax

    def probability_plot(
        self, fig: Figure = None, ax: Axes = None, distribution: str = "norm"
    ) -> tuple[Figure, Axes]:
        """Creates a probability plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        sp.stats.probplot(self.y, dist=distribution, plot=ax)
        ax.set_ylabel("Ordered Values")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_title(f"{distribution.capitalize()} Probability Plot")
        return fig, ax

    def box_cox_normality_plot(
        self, fig: Figure = None, ax: Axes = None
    ) -> tuple[Figure, Axes]:
        """Creates a Box-Cox normality plot of the data."""
        if fig is None or axes is None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        if axes.shape != (2, 2):
            raise ValueError("Axes must be an iterable of (2, 2) Axes objects.")

        sp.stats.boxcox_normplot(self.y, -2, 2, plot=axes[0, 1], N=200)
        transformed_y, maxlog = sp.stats.boxcox(self.y)
        axes[0, 0].hist(self.y, bins="auto")
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
        self,
        fig: Figure = None,
        ax: Axes = None,
        statistic: callable = np.mean,
    ) -> tuple[Figure, Axes]:
        """Creates a bootstrap plot of the data."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        result = sp.stats.bootstrap(
            (self.y,),
            statistic,
        )
        ax.hist(
            result.bootstrap_distribution,
            bins="auto",
            density=True,
            label=f"Variability of {statistic.__name__}: {result.standard_error:.3g}",
        )

        ax.set_xlabel(f"Bootstrap {statistic.__name__} values")
        ax.set_ylabel("Density")
        ax.set_title("Bootstrap Distribution")
        ax.legend()
        return fig, ax
