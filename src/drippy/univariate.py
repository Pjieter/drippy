"""Plotting functions for univariate models."""

from collections.abc import Callable
from collections.abc import Iterable
import numpy as np
import scipy as sp
from lmfit.models import LinearModel
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from drippy.utilities import get_figure_and_axes


class UnivariatePlotter:
    """A class for plotting univariate (y = constant + error) models."""

    def __init__(self, y: Iterable[float], x: Iterable[float]) -> None:
        """Initialize UnivariatePlotter with data.

        Args:
            y (Iterable[float]):
                Sequentially ordered data points
            x (Iterable[float]):
                Sequentially ordered "time" points. Can be any
                continuously changing variable.
        """
        self.y = np.asarray(y)
        self.x = np.asarray(x)
        if self.y.size == 0 or self.x.size == 0:
            error = "y and x cannot be empty"
            raise ValueError(error)
        if self.y.ndim != 1 or self.x.ndim != 1:
            error = "y and x must be 1-dimensional arrays"
            raise ValueError(error)
        if len(self.x) != len(self.y):
            error = (
                f"x and y must have the same length. "
                f"Got len(x)={len(self.x)}, len(y)={len(self.y)}."
            )
            raise ValueError(error)

    def auto_plot(self) -> None:
        """Creates automated diagnostic plots of the data."""
        # TODO: Implementation of the plotting logic goes here
        msg = "auto_plot is not yet implemented"
        raise NotImplementedError(msg)

    def sequence_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Creates a sequence plot showing data over index.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Axes | None, optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.

        Returns:
            tuple[Figure, Axes]:
                The figure and axes containing the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)
        ax.plot(self.x, self.y, label="Data")
        ax.legend()
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Sequence Plot")
        fig.tight_layout()
        return fig, ax

    def lag_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        lag: int = 1,
    ) -> tuple[Figure, Axes]:
        """Creates a lag plot of the data.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Axes | None, optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.
            lag (int, optional):
                Number of lags to use for the plot. Defaults to 1.

        Returns:
            tuple[Figure, Axes]:
                The figure and axes containing the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)

        y_lagged = self.y[lag:]
        y_original = self.y[:-lag]

        # Set color of point equal to its index to create a gradient effect
        colors = np.arange(len(y_lagged))
        scatter = ax.scatter(
            y_lagged,
            y_original,
            c=colors,
            cmap="viridis",
            label="Lag Plot",
        )
        fig.colorbar(scatter, ax=ax, label="Index")
        ax.set_xlabel(rf"Y$_{{i-{lag}}}$")
        ax.set_ylabel("Y$_{i}$")
        ax.set_title(f"Lag Plot (lag={lag})")
        fig.tight_layout()
        return fig, ax

    def histogram_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        bins: int | str = "auto",
    ) -> tuple[Figure, Axes]:
        """Creates a histogram plot of the data.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Axes | None, optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.
            bins (int | str, optional):
                Number of bins or method for computing bins.
                Defaults to "auto".

        Returns:
            tuple[Figure, Axes]:
                The figure and axes containing the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)

        ax.hist(self.y, bins=bins)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Y values")
        fig.tight_layout()
        return fig, ax

    def normal_probability_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        return_rsquared: bool = False,
    ) -> tuple[Figure, Axes] | tuple[Figure, Axes, float]:
        """Creates a normal probability plot of the data.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Axes | None, optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.
            return_rsquared (bool, optional):
                If True, returns R-squared value as third element.
                Defaults to False.

        Returns:
            tuple[Figure, Axes] | tuple[Figure, Axes, float]:
                The figure and axes containing the plot. If
                return_rsquared is True, also returns R-squared value.
        """
        fig, ax = get_figure_and_axes(fig, ax)

        _, (_, _, r_value) = sp.stats.probplot(
            self.y,
            dist="norm",
            plot=ax,
            rvalue=True,
        )
        ax.set_ylabel("Ordered Values")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_title("Normal Probability Plot")
        fig.tight_layout()
        if return_rsquared:
            return fig, ax, r_value**2
        return fig, ax

    def four_plot(
        self,
        fig: Figure | None = None,
        axes: Iterable[Axes] | None = None,
    ) -> tuple[Figure, Iterable[Axes]]:
        """Creates a 4-plot of the data.

        The 4-plot includes: sequence plot, lag plot, histogram,
        and normal probability plot.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            axes (Iterable[Axes] | None, optional):
                Array of matplotlib axes with shape (2, 2). If None,
                creates new axes. Defaults to None.

        Returns:
            tuple[Figure, Iterable[Axes]]:
                The figure and flattened array of axes containing the plots.

        Raises:
            ValueError:
                If axes is provided but does not have shape (2, 2).
        """
        if fig is None or axes is None:
            fig, axes = get_figure_and_axes(fig, None)
            axes = fig.subplots(2, 2)

        if axes.shape != (2, 2):
            error = "Axes must be an iterable of (2, 2) Axes objects."
            raise ValueError(error)

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
        fig: Figure | None = None,
        ax: Iterable[Axes] | None = None,
    ) -> tuple[Figure, Iterable[Axes]]:
        """Creates a probability plot correlation coefficient (PPCC) plot.

        Creates two subplots: a rough PPCC plot and a fine PPCC plot
        for Tukey-Lambda distribution shape parameter estimation.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Iterable[Axes] | None, optional):
                Array of matplotlib axes with shape (2,). If None,
                creates new axes. Defaults to None.

        Returns:
            tuple[Figure, Iterable[Axes]]:
                The figure and array of axes containing the plots.

        Raises:
            ValueError:
                If ax is provided but does not have shape (2,).
        """
        if fig is None or ax is None:
            fig, _ = get_figure_and_axes(fig, None)
            ax = fig.subplots(1, 2)

        if ax.shape != (2,):
            error = "Axes must be an iterable of 2 Axes objects."
            raise ValueError(error)

        rough_shape_values, rough_ppcc = sp.stats.ppcc_plot(
            self.y,
            -2,
            2,
            plot=ax[0],
        )
        rough_max_index = np.argmax(rough_ppcc)
        fine_shape_values, fine_ppcc = sp.stats.ppcc_plot(
            self.y,
            rough_shape_values[rough_max_index] - 0.5,
            rough_shape_values[rough_max_index] + 0.5,
            N=100,
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
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Creates a Weibull probability plot of the data.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Axes | None, optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.

        Returns:
            tuple[Figure, Axes]:
                The figure and axes containing the plot.

        Raises:
            ValueError:
                If any y values are not positive.
        """
        fig, ax = get_figure_and_axes(fig, ax)

        if not np.all(self.y > 0):
            error = "Weibull plot requires all y values to be positive."
            raise ValueError(error)

        # Create Weibull probability plot data
        n = len(self.y)
        p = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        weibull_probabilities = np.log(-np.log(1 - p))
        ordered_data = np.log(np.sort(self.y))

        # Fit line
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
                np.logspace(
                    min_percentage - 1,
                    -1,
                    num=int(-min_percentage) + 1,
                ),
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
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        distribution: str = "norm",
    ) -> tuple[Figure, Axes]:
        """Creates a probability plot for a specified distribution.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Axes | None, optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.
            distribution (str, optional):
                Distribution name for probability plot. Defaults to "norm".

        Returns:
            tuple[Figure, Axes]:
                The figure and axes containing the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)

        sp.stats.probplot(self.y, dist=distribution, plot=ax)
        ax.set_ylabel("Ordered Values")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_title(f"{distribution.capitalize()} Probability Plot")
        fig.tight_layout()
        return fig, ax

    def box_cox_normality_plot(
        self,
        fig: Figure | None = None,
        axes: Iterable[Axes] | None = None,
    ) -> tuple[Figure, Iterable[Axes]]:
        """Creates a Box-Cox normality plot of the data.

        Creates a 2x2 grid with: original histogram, Box-Cox normality
        plot, transformed histogram, and normal probability plot.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            axes (Iterable[Axes] | None, optional):
                Array of matplotlib axes with shape (2, 2). If None,
                creates new axes. Defaults to None.

        Returns:
            tuple[Figure, Iterable[Axes]]:
                The figure and array of axes containing the plots.

        Raises:
            ValueError:
                If axes is provided but does not have shape (2, 2).
        """
        if fig is None or axes is None:
            fig, axes = get_figure_and_axes(fig, None)
            axes = fig.subplots(2, 2)

        if axes.shape != (2, 2):
            error = "Axes must be an iterable of (2, 2) Axes objects."
            raise ValueError(error)

        sp.stats.boxcox_normplot(self.y, -2, 2, plot=axes[0, 1], N=200)
        transformed_y, maxlog = sp.stats.boxcox(self.y)
        axes[0, 0].hist(self.y, bins="auto")
        axes[0, 0].set_xlabel("Value")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title(
            "Histogram of Original (positive shifted) Y values",
        )
        axes[0, 1].set_title("Box-Cox Normality Plot")
        axes[0, 1].axvline(
            maxlog,
            color="r",
            linestyle="--",
            label=f"Max log={maxlog:.3g}",
        )
        axes[0, 1].legend()
        axes[1, 0].hist(transformed_y, bins="auto")
        axes[1, 0].set_xlabel("Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Histogram of transformed Y values")
        sp.stats.probplot(
            transformed_y,
            dist="norm",
            plot=axes[1, 1],
            rvalue=True,
        )
        axes[1, 1].set_title("Normal Probability Plot of Transformed Data")
        fig.tight_layout()
        return fig, axes

    def bootstrap_plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        statistic: Callable = np.mean,
    ) -> tuple[Figure, Axes]:
        """Creates a bootstrap distribution plot.

        Args:
            fig (Figure | None, optional):
                Matplotlib figure. If None, creates new figure.
                Defaults to None.
            ax (Axes | None, optional):
                Matplotlib axes. If None, creates new axes.
                Defaults to None.
            statistic (Callable, optional):
                Statistic function to bootstrap. Defaults to np.mean.

        Returns:
            tuple[Figure, Axes]:
                The figure and axes containing the plot.
        """
        fig, ax = get_figure_and_axes(fig, ax)
        result = sp.stats.bootstrap(
            (self.y,),
            statistic,
        )
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
