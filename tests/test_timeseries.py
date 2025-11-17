"""Tests for the drippy.timeseries module."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from drippy.timeseries import TimeSeriesPlotter

mpl.use("Agg")  # Use non-interactive backend for testing


@pytest.fixture
def simple_data():
    """Generate simple data for testing."""
    t = np.linspace(1, 10, 100)
    y = 2 * t + 3 + np.sin(t)  # Linear trend with sinusoidal component
    return t, y


@pytest.fixture
def plotter(simple_data):
    """Create a TimeSeriesPlotter instance."""
    t, y = simple_data
    return TimeSeriesPlotter(y=y, t=t)


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestTimeSeriesPlotterInitialization:
    """Tests for TimeSeriesPlotter initialization."""

    def test_initialization_with_lists(self):
        """Test initialization with list inputs."""
        t = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        plotter = TimeSeriesPlotter(y=y, t=t)

        assert isinstance(plotter.t, np.ndarray)
        assert isinstance(plotter.y, np.ndarray)
        assert len(plotter.t) == len(t)
        assert len(plotter.y) == len(y)

    def test_initialization_with_numpy_arrays(self, simple_data):
        """Test initialization with numpy array inputs."""
        t, y = simple_data
        plotter = TimeSeriesPlotter(y=y, t=t)

        assert isinstance(plotter.t, np.ndarray)
        assert isinstance(plotter.y, np.ndarray)
        assert len(plotter.t) == len(t)
        assert len(plotter.y) == len(y)

    def test_stores_data(self, simple_data):
        """Test that data is properly stored."""
        t, y = simple_data
        plotter = TimeSeriesPlotter(y=y, t=t)

        np.testing.assert_array_equal(plotter.t, t)
        np.testing.assert_array_equal(plotter.y, y)

    def test_raises_error_for_mismatched_lengths(self):
        """Test ValueError is raised for different length inputs."""
        t = [1, 2, 3]
        y = [1, 2, 3, 4, 5]
        with pytest.raises(
            ValueError, match=r"t and y must have the same length"
        ):
            TimeSeriesPlotter(y=y, t=t)


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
        """Test spectral_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.spectral_plot(fig=provided_fig, ax=provided_ax)

        assert fig is provided_fig
        assert ax is provided_ax


class TestSpectralPlot:
    """Tests for spectral_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that spectral_plot returns Figure and Axes objects."""
        fig, ax = plotter.spectral_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, plotter):
        """Test spectral_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.spectral_plot(fig=provided_fig, ax=provided_ax)

        assert fig is provided_fig
        assert ax is provided_ax

    def test_with_alarm_levels_enabled(self, plotter):
        """Test that spectral plot runs with alarm levels enabled."""
        _, ax = plotter.spectral_plot(alarm_levels=True)

        lines = ax.get_lines()
        assert len(lines) >= 1

    def test_with_alarm_levels_disabled(self, plotter):
        """Test that spectral plot runs with alarm levels disabled."""
        _, ax = plotter.spectral_plot(alarm_levels=False)

        lines = ax.get_lines()
        assert len(lines) >= 1

    def test_has_labels(self, plotter):
        """Test that spectral plot has proper axis labels."""
        _, ax = plotter.spectral_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        assert xlabel != ""
        assert ylabel != ""


class TestAutoCorrelationPlot:
    """Tests for auto_correlation_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that auto_correlation_plot returns Figure and Axes objects."""
        fig, ax = plotter.auto_correlation_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, plotter):
        """Test auto_correlation_plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.auto_correlation_plot(
            fig=provided_fig,
            ax=provided_ax,
        )

        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_plot_lines(self, plotter):
        """Test that autocorrelation plot creates line objects."""
        _, ax = plotter.auto_correlation_plot()

        lines = ax.get_lines()
        assert len(lines) > 0

    def test_has_labels(self, plotter):
        """Test that autocorrelation plot has proper axis labels."""
        _, ax = plotter.auto_correlation_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        assert xlabel != ""
        assert ylabel != ""


class TestComplexDemodulationPhasePlot:
    """Tests for complex_demodulation_phase_plot method."""

    def test_returns_figure_and_axes(self, plotter):
        """Test that plot returns Figure and Axes objects."""
        fig, ax = plotter.complex_demodulation_phase_plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_with_provided_fig_ax(self, plotter):
        """Test plot with user-provided figure and axes."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = plotter.complex_demodulation_phase_plot(
            fig=provided_fig,
            ax=provided_ax,
        )

        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_plot_lines(self, plotter):
        """Test that complex demodulation plot creates line objects."""
        _, ax = plotter.complex_demodulation_phase_plot()

        lines = ax.get_lines()
        assert len(lines) >= 1

    def test_has_labels(self, plotter):
        """Test complex demodulation plot has axis labels and title."""
        _, ax = plotter.complex_demodulation_phase_plot()

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = ax.get_title()

        assert xlabel != ""
        assert ylabel != ""
        assert title != ""


class TestEdgeCases:
    """Tests for edge cases and data handling."""

    def test_empty_data(self):
        """Test initialization with empty data."""
        t = np.array([])
        y = np.array([])
        plotter = TimeSeriesPlotter(y=y, t=t)

        assert len(plotter.t) == 0
        assert len(plotter.y) == 0

    def test_single_data_point(self):
        """Test initialization with a single data point."""
        t = np.array([1.0])
        y = np.array([2.0])
        plotter = TimeSeriesPlotter(y=y, t=t)

        assert len(plotter.t) == 1
        assert len(plotter.y) == 1

    def test_large_dataset(self):
        """Test initialization with a large dataset."""
        t = np.linspace(0, 1000, 10000)
        y = np.sin(t)
        plotter = TimeSeriesPlotter(y=y, t=t)

        assert len(plotter.t) == len(t)
        assert len(plotter.y) == len(y)

    def test_negative_values(self):
        """Test with negative values in data."""
        t = np.array([1, 2, 3, 4, 5])
        y = np.array([-5, -3, -1, 1, 3])
        plotter = TimeSeriesPlotter(y=y, t=t)

        assert len(plotter.t) == len(t)
        assert len(plotter.y) == len(y)
