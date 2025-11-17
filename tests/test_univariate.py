"""Tests for the drippy.univariate module."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from drippy.univariate import UnivariatePlotter

mpl.use("Agg")  # Use non-interactive backend for testing


@pytest.fixture
def simple_data():
    """Generate simple data for testing."""
    x = np.linspace(1, 10, 100)
    y = 5 + np.random.default_rng(42).normal(0, 1, 100)  # Constant + noise
    return x, y


@pytest.fixture
def positive_data():
    """Generate positive data for Weibull and Box-Cox tests."""
    x = np.linspace(1, 10, 50)
    y = np.abs(5 + np.random.default_rng(42).normal(0, 1, 50)) + 1
    return x, y


@pytest.fixture
def plotter(simple_data):
    """Create a UnivariatePlotter instance."""
    x, y = simple_data
    return UnivariatePlotter(y=y, x=x)


@pytest.fixture
def positive_plotter(positive_data):
    """Create a UnivariatePlotter instance with positive data."""
    x, y = positive_data
    return UnivariatePlotter(y=y, x=x)


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestUnivariatePlotterInitialization:
    """Tests for UnivariatePlotter initialization."""

    def test_initialization_with_lists(self):
        """Test initialization with list inputs."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        plotter = UnivariatePlotter(y=y, x=x)

        assert isinstance(plotter.x, np.ndarray)
        assert isinstance(plotter.y, np.ndarray)
        assert len(plotter.x) == len(x)
        assert len(plotter.y) == len(y)

    def test_initialization_with_numpy_arrays(self, simple_data):
        """Test initialization with numpy array inputs."""
        x, y = simple_data
        plotter = UnivariatePlotter(y=y, x=x)

        assert isinstance(plotter.x, np.ndarray)
        assert isinstance(plotter.y, np.ndarray)
        assert len(plotter.x) == len(x)
        assert len(plotter.y) == len(y)

    def test_stores_data(self, simple_data):
        """Test that data is properly stored."""
        x, y = simple_data
        plotter = UnivariatePlotter(y=y, x=x)

        np.testing.assert_array_equal(plotter.x, x)
        np.testing.assert_array_equal(plotter.y, y)

    def test_raises_error_for_mismatched_lengths(self):
        """Test ValueError is raised for different length inputs."""
        x = [1, 2, 3]
        y = [1, 2, 3, 4, 5]
        with pytest.raises(
            ValueError,
            match=r"x and y must have the same length",
        ):
            UnivariatePlotter(y=y, x=x)

    def test_raises_error_for_empty_data(self):
        """Test ValueError is raised for empty inputs."""
        x = []
        y = []
        with pytest.raises(ValueError, match=r"y and x cannot be empty"):
            UnivariatePlotter(y=y, x=x)

    def test_raises_error_for_empty_y(self):
        """Test ValueError is raised when y is empty."""
        x = [1, 2, 3]
        y = []
        with pytest.raises(ValueError, match=r"y and x cannot be empty"):
            UnivariatePlotter(y=y, x=x)

    def test_raises_error_for_empty_x(self):
        """Test ValueError is raised when x is empty."""
        x = []
        y = [1, 2, 3]
        with pytest.raises(ValueError, match=r"y and x cannot be empty"):
            UnivariatePlotter(y=y, x=x)

    def test_raises_error_for_multidimensional_arrays(self):
        """Test ValueError is raised for multi-dimensional inputs."""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        with pytest.raises(
            ValueError,
            match=r"y and x must be 1-dimensional arrays",
        ):
            UnivariatePlotter(y=y, x=x)

    def test_raises_error_for_multidimensional_y(self):
        """Test ValueError is raised when y is multi-dimensional."""
        x = np.array([1, 2, 3, 4])
        y = np.array([[5, 6], [7, 8]])
        with pytest.raises(
            ValueError,
            match=r"y and x must be 1-dimensional arrays",
        ):
            UnivariatePlotter(y=y, x=x)

    def test_raises_error_for_multidimensional_x(self):
        """Test ValueError is raised when x is multi-dimensional."""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([5, 6, 7, 8])
        with pytest.raises(
            ValueError,
            match=r"y and x must be 1-dimensional arrays",
        ):
            UnivariatePlotter(y=y, x=x)


class TestSequencePlot:
    """Tests for sequence_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that sequence_plot returns Figure and Axes objects."""
        fig, ax = plotter.sequence_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_creates_plot_lines(self, plotter):
        """Test that sequence plot creates line objects."""
        _, ax = plotter.sequence_plot()

        lines = ax.get_lines()
        assert len(lines) >= 1

    def test_has_labels(self, plotter):
        """Test that sequence plot has axis labels and title."""
        _, ax = plotter.sequence_plot()

        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""

    def test_with_provided_fig_ax(self, plotter):
        """Test sequence_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.sequence_plot(fig=provided_fig, ax=provided_ax)

        assert fig is provided_fig
        assert ax is provided_ax

    def test_plots_correct_data(self, plotter):
        """Test that sequence plot contains correct data."""
        _, ax = plotter.sequence_plot()

        lines = ax.get_lines()
        x_data, y_data = lines[0].get_data()

        np.testing.assert_array_equal(x_data, plotter.x)
        np.testing.assert_array_equal(y_data, plotter.y)


class TestLagPlot:
    """Tests for lag_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that lag_plot returns Figure and Axes objects."""
        fig, ax = plotter.lag_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, plotter):
        """Test lag_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.lag_plot(fig=provided_fig, ax=provided_ax)

        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_scatter_plot(self, plotter):
        """Test that lag plot creates scatter plot."""
        _, ax = plotter.lag_plot()

        collections = ax.collections
        assert len(collections) >= 1

    def test_has_labels(self, plotter):
        """Test that lag plot has proper axis labels."""
        _, ax = plotter.lag_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        assert xlabel != ""
        assert ylabel != ""

    def test_with_different_lags(self, plotter):
        """Test lag plot with different lag values."""
        for lag in [1, 2, 5]:
            _, ax = plotter.lag_plot(lag=lag)
            assert ax.get_title() == f"Lag Plot (lag={lag})"

    def test_lag_plot_data_length(self, plotter):
        """Test that lag plot has correct data length."""
        lag = 3
        _, ax = plotter.lag_plot(lag=lag)

        collections = ax.collections
        scatter_data = collections[0].get_offsets()

        expected_length = len(plotter.y) - lag
        assert len(scatter_data) == expected_length

    def test_has_colorbar(self, plotter):
        """Test that lag plot has a colorbar."""
        fig, _ = plotter.lag_plot()

        # Check for colorbar by looking at figure axes
        # Main plot + colorbar
        assert len(fig.axes) == 2  # noqa: PLR2004


class TestHistogramPlot:
    """Tests for histogram_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that histogram_plot returns Figure and Axes objects."""
        fig, ax = plotter.histogram_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, plotter):
        """Test histogram_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.histogram_plot(fig=provided_fig, ax=provided_ax)

        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_histogram(self, plotter):
        """Test that histogram plot creates patches."""
        _, ax = plotter.histogram_plot()

        patches = ax.patches
        assert len(patches) > 0

    def test_has_labels(self, plotter):
        """Test that histogram plot has proper axis labels."""
        _, ax = plotter.histogram_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()

        assert xlabel != ""
        assert ylabel != ""
        assert title != ""

    def test_with_different_bins(self, plotter):
        """Test histogram plot with different bin specifications."""
        for bins in [5, 10, "auto"]:
            _, ax = plotter.histogram_plot(bins=bins)
            patches = ax.patches
            assert len(patches) > 0

    def test_with_integer_bins(self, plotter):
        """Test histogram plot with specific integer bins."""
        _, ax = plotter.histogram_plot(bins=10)

        patches = ax.patches
        # 10 bins specified
        assert len(patches) == 10  # noqa: PLR2004


class TestNormalProbabilityPlot:
    """Tests for normal_probability_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that normal_probability_plot returns Figure and Axes."""
        fig, ax = plotter.normal_probability_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, plotter):
        """Test normal_probability_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.normal_probability_plot(
            fig=provided_fig,
            ax=provided_ax,
        )

        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_plot_lines(self, plotter):
        """Test that normal probability plot creates line objects."""
        _, ax = plotter.normal_probability_plot()

        lines = ax.get_lines()
        assert len(lines) >= 1

    def test_has_labels(self, plotter):
        """Test that normal probability plot has proper axis labels."""
        _, ax = plotter.normal_probability_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()

        assert xlabel != ""
        assert ylabel != ""
        assert title != ""

    def test_returns_rsquared_when_requested(self, plotter):
        """Test that R-squared is returned when requested."""
        result = plotter.normal_probability_plot(return_rsquared=True)

        # Should return fig, ax, and rsquared
        assert len(result) == 3  # noqa: PLR2004
        fig, ax, rsquared = result

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert isinstance(rsquared, float)
        assert 0 <= rsquared <= 1

    def test_returns_tuple_of_two_by_default(self, plotter):
        """Test that only fig and ax are returned by default."""
        result = plotter.normal_probability_plot()

        # Should return only fig and ax (not rsquared)
        assert len(result) == 2  # noqa: PLR2004
        assert isinstance(result[0], Figure)
        assert isinstance(result[1], Axes)


class TestFourPlot:
    """Tests for four_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that four_plot returns Figure and Axes objects."""
        fig, axes = plotter.four_plot()

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Four-plot returns 4 axes
        assert len(axes) == 4  # noqa: PLR2004

    def test_creates_four_subplots(self, plotter):
        """Test that four_plot creates 4 subplots."""
        fig, axes = plotter.four_plot()

        # Four-plot creates 4 main subplots (may have colorbar as extra)
        assert len(fig.axes) >= 4  # noqa: PLR2004
        assert len(axes) == 4  # noqa: PLR2004
        for ax in axes:
            assert isinstance(ax, Axes)

    def test_all_subplots_have_content(self, plotter):
        """Test that all subplots have content."""
        _, axes = plotter.four_plot()

        for ax in axes:
            # Check that each subplot has either lines, patches, or collections
            has_content = (
                len(ax.get_lines()) > 0
                or len(ax.patches) > 0
                or len(ax.collections) > 0
            )
            assert has_content

    def test_raises_error_for_wrong_axes_shape(self, plotter):
        """Test ValueError for axes with wrong shape."""
        fig, axes = plt.subplots(1, 3)

        with pytest.raises(
            ValueError,
            match=r"Axes must be an iterable of \(2, 2\) Axes objects\.",
        ):
            plotter.four_plot(fig=fig, axes=axes)

    def test_with_provided_fig_and_axes(self, plotter):
        """Test four_plot with user-provided figure and axes."""
        provided_fig, provided_axes = plt.subplots(2, 2)
        fig, axes = plotter.four_plot(fig=provided_fig, axes=provided_axes)

        assert fig is provided_fig
        # Check that the returned axes are from the provided axes
        for i, ax in enumerate(axes):
            assert ax is provided_axes.flatten()[i]


class TestPPCCPlot:
    """Tests for ppcc_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that ppcc_plot returns Figure and Axes objects."""
        fig, axes = plotter.ppcc_plot()

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # PPCC plot has rough and fine subplots
        assert len(axes) == 2  # noqa: PLR2004

    def test_creates_two_subplots(self, plotter):
        """Test that ppcc_plot creates 2 subplots."""
        _, axes = plotter.ppcc_plot()

        # PPCC has rough and fine plots
        assert len(axes) == 2  # noqa: PLR2004
        for ax in axes:
            assert isinstance(ax, Axes)

    def test_subplots_have_titles(self, plotter):
        """Test that both subplots have titles."""
        _, axes = plotter.ppcc_plot()

        assert axes[0].get_title() == "Rough PPCC Plot"
        assert axes[1].get_title() == "Fine PPCC Plot"

    def test_subplots_have_lines(self, plotter):
        """Test that both subplots have line content."""
        _, axes = plotter.ppcc_plot()

        for ax in axes:
            lines = ax.get_lines()
            assert len(lines) > 0

    def test_raises_error_for_wrong_axes_shape(self, plotter):
        """Test ValueError for axes with wrong shape."""
        fig, axes = plt.subplots(1, 3)

        with pytest.raises(
            ValueError,
            match=r"Axes must be an iterable of 2 Axes objects\.",
        ):
            plotter.ppcc_plot(fig=fig, ax=axes)

    def test_with_provided_fig_and_axes(self, plotter):
        """Test ppcc_plot with user-provided figure and axes."""
        provided_fig, provided_axes = plt.subplots(1, 2)
        fig, axes = plotter.ppcc_plot(fig=provided_fig, ax=provided_axes)

        assert fig is provided_fig
        for i, ax in enumerate(axes):
            assert ax is provided_axes[i]


class TestWeibullPlot:
    """Tests for weibull_plot method."""

    def test_returns_figure_and_axes(self, positive_plotter):
        """Test that weibull_plot returns Figure and Axes objects."""
        fig, ax = positive_plotter.weibull_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, positive_plotter):
        """Test weibull_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = positive_plotter.weibull_plot(
            fig=provided_fig,
            ax=provided_ax,
        )

        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_plot_lines(self, positive_plotter):
        """Test that Weibull plot creates line objects."""
        _, ax = positive_plotter.weibull_plot()

        lines = ax.get_lines()
        # At least data points and fit line
        assert len(lines) >= 2  # noqa: PLR2004

    def test_has_labels(self, positive_plotter):
        """Test that Weibull plot has proper axis labels."""
        _, ax = positive_plotter.weibull_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()

        assert xlabel != ""
        assert ylabel != ""
        assert title != ""

    def test_raises_error_for_non_positive_values(self):
        """Test ValueError for non-positive y values."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, -2, 3, 4, 5])  # Contains negative value
        plotter = UnivariatePlotter(y=y, x=x)

        with pytest.raises(
            ValueError,
            match=r"Weibull plot requires all y values to be positive\.",
        ):
            plotter.weibull_plot()

    def test_raises_error_for_zero_values(self):
        """Test ValueError for zero y values."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 0, 3, 4, 5])  # Contains zero
        plotter = UnivariatePlotter(y=y, x=x)

        with pytest.raises(
            ValueError,
            match=r"Weibull plot requires all y values to be positive\.",
        ):
            plotter.weibull_plot()

    def test_has_reference_lines(self, positive_plotter):
        """Test that Weibull plot has reference lines."""
        _, ax = positive_plotter.weibull_plot()

        # Count horizontal and vertical lines
        lines = ax.get_lines()
        # Should have data points, fit line, and reference lines (min 3)
        assert len(lines) >= 2  # noqa: PLR2004


class TestProbabilityPlot:
    """Tests for probability_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that probability_plot returns Figure and Axes objects."""
        fig, ax = plotter.probability_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, plotter):
        """Test probability_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.probability_plot(fig=provided_fig, ax=provided_ax)

        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_plot_lines(self, plotter):
        """Test that probability plot creates line objects."""
        _, ax = plotter.probability_plot()

        lines = ax.get_lines()
        assert len(lines) >= 1

    def test_has_labels(self, plotter):
        """Test that probability plot has proper axis labels."""
        _, ax = plotter.probability_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()

        assert xlabel != ""
        assert ylabel != ""
        assert title != ""

    def test_with_different_distributions(self, plotter):
        """Test probability plot with different distributions."""
        for dist in ["norm", "uniform", "expon"]:
            _, ax = plotter.probability_plot(distribution=dist)
            title = ax.get_title()
            assert dist.capitalize() in title

    def test_default_distribution_is_normal(self, plotter):
        """Test that default distribution is normal."""
        _, ax = plotter.probability_plot()

        title = ax.get_title()
        assert "Norm" in title


class TestBoxCoxNormalityPlot:
    """Tests for box_cox_normality_plot method."""

    def test_returns_figure_and_axes(self, positive_plotter):
        """Test that box_cox_normality_plot returns Figure and Axes."""
        fig, axes = positive_plotter.box_cox_normality_plot()

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 2)

    def test_creates_four_subplots(self, positive_plotter):
        """Test that box_cox_normality_plot creates 4 subplots."""
        fig, axes = positive_plotter.box_cox_normality_plot()

        # Box-Cox creates 2x2 grid = 4 subplots (unless a user supplies
        # additional axes in a figure).
        assert len(fig.axes) >= 4  # noqa: PLR2004
        for ax in axes.flatten():
            assert isinstance(ax, Axes)

    def test_all_subplots_have_content(self, positive_plotter):
        """Test that all subplots have content."""
        _, axes = positive_plotter.box_cox_normality_plot()

        for ax in axes.flatten():
            # Check that each subplot has either lines or patches
            has_content = len(ax.get_lines()) > 0 or len(ax.patches) > 0
            assert has_content

    def test_all_subplots_have_titles(self, positive_plotter):
        """Test that all subplots have titles."""
        _, axes = positive_plotter.box_cox_normality_plot()

        for ax in axes.flatten():
            title = ax.get_title()
            assert title != ""

    def test_raises_error_for_wrong_axes_shape(self, positive_plotter):
        """Test ValueError for axes with wrong shape."""
        fig, axes = plt.subplots(1, 3)

        with pytest.raises(
            ValueError,
            match=r"Axes must be an iterable of \(2, 2\) Axes objects\.",
        ):
            positive_plotter.box_cox_normality_plot(fig=fig, axes=axes)

    def test_with_provided_fig_and_axes(self, positive_plotter):
        """Test box_cox_normality_plot with user-provided figure and axes."""
        provided_fig, provided_axes = plt.subplots(2, 2, figsize=(10, 10))
        fig, axes = positive_plotter.box_cox_normality_plot(
            fig=provided_fig,
            axes=provided_axes,
        )

        assert fig is provided_fig
        # Check that the returned axes are from the provided axes
        for i, ax in enumerate(axes.flatten()):
            assert ax is provided_axes.flatten()[i]


class TestBootstrapPlot:
    """Tests for bootstrap_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that bootstrap_plot returns Figure and Axes objects."""
        fig, ax = plotter.bootstrap_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, plotter):
        """Test bootstrap_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.bootstrap_plot(fig=provided_fig, ax=provided_ax)

        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_histogram(self, plotter):
        """Test that bootstrap plot creates histogram."""
        _, ax = plotter.bootstrap_plot()

        patches = ax.patches
        assert len(patches) > 0

    def test_has_labels(self, plotter):
        """Test that bootstrap plot has proper axis labels."""
        _, ax = plotter.bootstrap_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()

        assert xlabel != ""
        assert ylabel != ""
        assert title != ""

    def test_with_different_statistics(self, plotter):
        """Test bootstrap plot with different statistics."""
        for stat in [np.mean, np.median, np.std]:
            _, ax = plotter.bootstrap_plot(statistic=stat)
            xlabel = ax.get_xlabel()
            assert stat.__name__ in xlabel

    def test_default_statistic_is_mean(self, plotter):
        """Test that default statistic is mean."""
        _, ax = plotter.bootstrap_plot()

        xlabel = ax.get_xlabel()
        assert "mean" in xlabel


class TestEdgeCases:
    """Tests for edge cases and data handling."""

    def test_single_data_point(self):
        """Test initialization with a single data point."""
        x = np.array([1.0])
        y = np.array([2.0])
        plotter = UnivariatePlotter(y=y, x=x)

        assert len(plotter.x) == 1
        assert len(plotter.y) == 1

    def test_large_dataset(self):
        """Test initialization with a large dataset."""
        x = np.linspace(0, 1000, 10000)
        y = 5 + np.random.default_rng(42).normal(0, 1, 10000)
        plotter = UnivariatePlotter(y=y, x=x)

        assert len(plotter.x) == len(x)
        assert len(plotter.y) == len(y)

    def test_negative_values(self):
        """Test with negative values in data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([-5, -3, -1, 1, 3])
        plotter = UnivariatePlotter(y=y, x=x)

        assert len(plotter.x) == len(x)
        assert len(plotter.y) == len(y)

    def test_constant_data(self):
        """Test with constant y values."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 5, 5, 5, 5])
        plotter = UnivariatePlotter(y=y, x=x)

        fig, ax = plotter.sequence_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_very_small_values(self):
        """Test with very small y values."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        plotter = UnivariatePlotter(y=y, x=x)

        assert len(plotter.y) == len(y)

    def test_very_large_values(self):
        """Test with very large y values."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        plotter = UnivariatePlotter(y=y, x=x)

        assert len(plotter.y) == len(y)

    def test_mixed_positive_negative_values(self):
        """Test with mixed positive and negative values."""
        x = np.linspace(1, 10, 50)
        y = np.sin(x) * 10  # Oscillating between positive and negative
        plotter = UnivariatePlotter(y=y, x=x)

        fig, _ = plotter.sequence_plot()
        assert isinstance(fig, Figure)

    def test_two_data_points(self):
        """Test with only two data points."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        plotter = UnivariatePlotter(y=y, x=x)

        fig, _ = plotter.sequence_plot()
        assert isinstance(fig, Figure)

    def test_lag_plot_with_large_lag(self):
        """Test lag plot with lag close to data length."""
        x = np.arange(10)
        y = np.arange(10)
        plotter = UnivariatePlotter(y=y, x=x)

        # Lag of 9 should leave only 1 data point
        _, ax = plotter.lag_plot(lag=9)
        assert isinstance(ax, Axes)

    def test_histogram_with_few_unique_values(self):
        """Test histogram with few unique values."""
        x = np.arange(20)
        y = np.array([1, 2, 3] * 6 + [1, 2])  # Only 3 unique values
        plotter = UnivariatePlotter(y=y, x=x)

        fig, _ = plotter.histogram_plot()
        assert isinstance(fig, Figure)


class TestIntegration:
    """Integration tests for multiple methods."""

    def test_multiple_plots_same_figure(self, plotter):
        """Test creating multiple plots on the same figure."""
        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        fig_ret1, ax1_out = plotter.sequence_plot(fig=fig, ax=ax1)
        fig_ret2, ax2_out = plotter.lag_plot(fig=fig, ax=ax2)
        fig_ret3, ax3_out = plotter.histogram_plot(fig=fig, ax=ax3)
        fig_ret4, ax4_out = plotter.normal_probability_plot(fig=fig, ax=ax4)

        # All should return the same figure
        assert fig_ret1 is fig_ret2 is fig_ret3 is fig_ret4 is fig

        # Each should return the correct axes
        assert ax1_out is ax1
        assert ax2_out is ax2
        assert ax3_out is ax3
        assert ax4_out is ax4

    def test_sequential_plotting_calls(self, plotter):
        """Test multiple sequential plotting calls."""
        plots = [
            plotter.sequence_plot(),
            plotter.lag_plot(),
            plotter.histogram_plot(),
            plotter.normal_probability_plot(),
            plotter.probability_plot(),
        ]

        for fig, ax in plots:
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

    def test_data_consistency_across_plots(self, plotter):
        """Test that data remains consistent across different plot calls."""
        original_y = plotter.y.copy()
        original_x = plotter.x.copy()

        # Create multiple plots
        plotter.sequence_plot()
        plotter.lag_plot()
        plotter.histogram_plot()
        plotter.normal_probability_plot()

        # Data should not change
        np.testing.assert_array_equal(plotter.y, original_y)
        np.testing.assert_array_equal(plotter.x, original_x)

    def test_four_plot_versus_individual_plots(self, plotter):
        """Test that four_plot produces same result as individual plots."""
        fig1, _ = plotter.four_plot()

        fig2, axes2 = plt.subplots(2, 2)
        plotter.sequence_plot(fig=fig2, ax=axes2[0, 0])
        plotter.lag_plot(fig=fig2, ax=axes2[0, 1])
        plotter.histogram_plot(fig=fig2, ax=axes2[1, 0])
        plotter.normal_probability_plot(fig=fig2, ax=axes2[1, 1])

        # Both figures should have at least 4 main plot axes
        # (may have additional colorbar axes)
        assert len(fig1.axes) >= 4  # noqa: PLR2004
        assert len(fig2.axes) >= 4  # noqa: PLR2004

    def test_positive_data_workflow(self, positive_plotter):
        """Test complete workflow with positive data."""
        # Should be able to create all plots including Weibull
        plots = [
            positive_plotter.sequence_plot(),
            positive_plotter.histogram_plot(),
            positive_plotter.weibull_plot(),
            positive_plotter.box_cox_normality_plot(),
        ]

        for result in plots:
            # Check tuple length to determine plot type
            if len(result) == 2:  # noqa: PLR2004
                fig, _ = result
                assert isinstance(fig, Figure)
            else:
                fig, _ = result
                assert isinstance(fig, Figure)


class TestDataTypes:
    """Tests for different data types and conversions."""

    def test_with_python_lists(self):
        """Test with Python lists."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        plotter = UnivariatePlotter(y=y, x=x)

        assert isinstance(plotter.x, np.ndarray)
        assert isinstance(plotter.y, np.ndarray)

    def test_with_tuples(self):
        """Test with tuples."""
        x = (1, 2, 3, 4, 5)
        y = (2, 4, 6, 8, 10)
        plotter = UnivariatePlotter(y=y, x=x)

        assert isinstance(plotter.x, np.ndarray)
        assert isinstance(plotter.y, np.ndarray)

    def test_with_integer_arrays(self):
        """Test with integer numpy arrays."""
        x = np.array([1, 2, 3, 4, 5], dtype=int)
        y = np.array([2, 4, 6, 8, 10], dtype=int)
        plotter = UnivariatePlotter(y=y, x=x)

        fig, _ = plotter.sequence_plot()
        assert isinstance(fig, Figure)

    def test_with_float_arrays(self):
        """Test with float numpy arrays."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
        plotter = UnivariatePlotter(y=y, x=x)

        fig, _ = plotter.sequence_plot()
        assert isinstance(fig, Figure)

    def test_with_mixed_types(self):
        """Test with mixed integer and float values."""
        x = [1, 2.5, 3, 4.5, 5]
        y = [2.0, 4, 6.5, 8, 10.0]
        plotter = UnivariatePlotter(y=y, x=x)

        assert plotter.x.dtype in [np.float64, np.float32]
        assert plotter.y.dtype in [np.float64, np.float32]


class TestParameterValidation:
    """Tests for parameter validation in plotting methods."""

    def test_histogram_bins_validation(self, plotter):
        """Test that histogram accepts valid bin specifications."""
        valid_bins = [5, 10, 20, "auto", "sqrt", "sturges"]

        for bins in valid_bins:
            fig, _ = plotter.histogram_plot(bins=bins)
            assert isinstance(fig, Figure)

    def test_histogram_with_negative_bins(self, plotter):
        """Test histogram with negative number of bins."""
        with pytest.raises(ValueError):
            plotter.histogram_plot(bins=-5)

    def test_histogram_with_zero_bins(self, plotter):
        """Test histogram with zero bins."""
        with pytest.raises(ValueError):
            plotter.histogram_plot(bins=0)

    def test_histogram_with_single_bin(self, plotter):
        """Test histogram with single bin."""
        fig, ax = plotter.histogram_plot(bins=1)
        assert isinstance(fig, Figure)
        assert len(ax.patches) == 1

    def test_lag_plot_with_small_lag(self, plotter):
        """Test lag plot behavior with small lag."""
        # With lag=1, should have n-1 points
        _, ax = plotter.lag_plot(lag=1)
        collections = ax.collections
        scatter_data = collections[0].get_offsets()
        assert len(scatter_data) == len(plotter.y) - 1

    def test_lag_plot_with_negative_lag(self, plotter):
        """Test lag plot with negative lag value."""
        with pytest.raises(
            ValueError,
            match=r"Lag must be a positive integer",
        ):
            plotter.lag_plot(lag=-1)

    def test_lag_plot_with_zero_lag(self, plotter):
        """Test lag plot with zero lag value."""
        with pytest.raises(
            ValueError,
            match=r"Lag must be a positive integer",
        ):
            plotter.lag_plot(lag=0)

    def test_lag_plot_with_lag_equal_to_data_length(self, plotter):
        """Test lag plot with lag equal to data length."""
        with pytest.raises(
            ValueError,
            match=r"Lag must be less than the length of the data",
        ):
            plotter.lag_plot(lag=len(plotter.y))

    def test_lag_plot_with_lag_exceeding_data_length(self, plotter):
        """Test lag plot with lag exceeding data length."""
        with pytest.raises(
            ValueError,
            match=r"Lag must be less than the length of the data",
        ):
            plotter.lag_plot(lag=len(plotter.y) + 10)

    def test_lag_plot_with_float_lag(self, plotter):
        """Test lag plot with float lag value (should convert to int)."""
        # Should accept float and convert to int
        fig, ax = plotter.lag_plot(lag=2.7)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_probability_plot_with_invalid_distribution(self, plotter):
        """Test probability plot with invalid distribution name."""
        # scipy.stats.probplot raises ValueError for invalid distribution
        with pytest.raises(
            ValueError,
            match=r"is not a valid distribution name",
        ):
            plotter.probability_plot(distribution="invalid_dist")

    def test_bootstrap_plot_with_negative_n_bootstrap(self, plotter):
        """Test bootstrap plot with negative number of bootstrap samples."""
        with pytest.raises(
            ValueError,
            match=r"Number of bootstrap samples must be positive",
        ):
            plotter.bootstrap_plot(n_bootstrap=-100)

    def test_bootstrap_plot_with_zero_n_bootstrap(self, plotter):
        """Test bootstrap plot with zero bootstrap samples."""
        with pytest.raises(
            ValueError,
            match=r"Number of bootstrap samples must be positive",
        ):
            plotter.bootstrap_plot(n_bootstrap=0)

    def test_bootstrap_plot_with_single_bootstrap(self, plotter):
        """Test bootstrap plot with single bootstrap sample."""
        fig, ax = plotter.bootstrap_plot(n_bootstrap=1)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_bootstrap_plot_with_non_callable_statistic(self, plotter):
        """Test bootstrap plot with non-callable statistic."""
        with pytest.raises(
            TypeError,
            match=r"Statistic must be callable",
        ):
            plotter.bootstrap_plot(statistic="mean")

    def test_ppcc_plot_with_invalid_rough_range(self, positive_plotter):
        """Test PPCC plot with invalid rough range."""
        with pytest.raises(
            ValueError,
            match=r"Rough range must contain exactly 2 elements",
        ):
            positive_plotter.ppcc_plot(rough_range=(0.1, 0.5, 0.9))

    def test_ppcc_plot_with_inverted_rough_range(self, positive_plotter):
        """Test PPCC plot with inverted rough range."""
        with pytest.raises(
            ValueError,
            match=r"Rough range must be \(min, max\) with min < max",
        ):
            positive_plotter.ppcc_plot(rough_range=(0.9, 0.1))

    def test_ppcc_plot_with_equal_rough_range(self, positive_plotter):
        """Test PPCC plot with equal values in rough range."""
        with pytest.raises(
            ValueError,
            match=r"Rough range must be \(min, max\) with min < max",
        ):
            positive_plotter.ppcc_plot(rough_range=(0.5, 0.5))

    def test_ppcc_plot_with_negative_n_points(self, positive_plotter):
        """Test PPCC plot with negative number of points."""
        with pytest.raises(
            ValueError,
            match=r"Number of points must be positive",
        ):
            positive_plotter.ppcc_plot(n_rough=-10)

    def test_ppcc_plot_with_zero_n_points(self, positive_plotter):
        """Test PPCC plot with zero points."""
        with pytest.raises(
            ValueError,
            match=r"Number of points must be positive",
        ):
            positive_plotter.ppcc_plot(n_rough=0)


class TestMemoryAndPerformance:
    """Tests for memory management and performance."""

    def test_large_dataset_memory(self):
        """Test that large datasets don't cause memory issues."""
        x = np.linspace(0, 1000, 100000)
        y = 5 + np.random.default_rng(42).normal(0, 1, 100000)
        plotter = UnivariatePlotter(y=y, x=x)

        # Should be able to create plots without memory issues
        fig, _ = plotter.sequence_plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

        fig, _ = plotter.histogram_plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_repeated_plotting(self, plotter):
        """Test that repeated plotting doesn't cause issues."""
        for _ in range(10):
            fig, _ = plotter.sequence_plot()
            plt.close(fig)

        # Should still be able to plot
        fig, _ = plotter.sequence_plot()
        assert isinstance(fig, Figure)
