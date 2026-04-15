"""Tests for the drippy.timeseries module."""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mpl.use("Agg")

import drippy.timeseries as ts
import drippy.univariate as uv
from drippy.data import EDAData

# --- Constants ---

MIN_PHASE_PLOT_LINES = 2


# --- Fixtures ---


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def timeseries_data():
    t = np.linspace(1, 10, 100)
    y = 2 * t + 3 + np.sin(t)
    return EDAData(y, t=t)


@pytest.fixture
def univariate_data():
    """No t — for testing functions that require t."""
    return EDAData(np.random.default_rng(42).normal(size=50))


# --- run_sequence_plot (shared from univariate) ---


class TestRunSequencePlotShared:
    """Tests for run_sequence_plot."""

    def test_imported_from_univariate(self):
        """Verify run_sequence_plot is imported from univariate."""
        assert ts.run_sequence_plot is uv.run_sequence_plot

    def test_uses_t_when_provided(self, timeseries_data):
        """Verify that t is used as x-axis when provided."""
        _, ax = ts.run_sequence_plot(timeseries_data)
        lines = ax.get_lines()
        x_data, _ = lines[0].get_data()
        np.testing.assert_array_equal(x_data, timeseries_data.t)


# --- spectral_plot ---


class TestSpectralPlot:
    """Tests for spectral_plot."""

    def test_returns_figure_and_axes(self, timeseries_data):
        """Verify spectral_plot returns Figure and Axes."""
        fig, ax = ts.spectral_plot(timeseries_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_t(self, univariate_data):
        """Verify spectral_plot raises ValueError when t is missing."""
        with pytest.raises(ValueError, match="requires t"):
            ts.spectral_plot(univariate_data)

    def test_custom_fig_ax(self, timeseries_data):
        """Verify spectral_plot respects custom fig and ax."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = ts.spectral_plot(
            timeseries_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_alarm_levels_enabled(self, timeseries_data):
        """Verify spectral_plot works with alarm levels enabled."""
        _, ax = ts.spectral_plot(timeseries_data, alarm_levels=True)
        assert len(ax.get_lines()) >= 1

    def test_alarm_levels_disabled(self, timeseries_data):
        """Verify spectral_plot works with alarm levels disabled."""
        _, ax = ts.spectral_plot(timeseries_data, alarm_levels=False)
        assert len(ax.get_lines()) >= 1

    def test_has_labels(self, timeseries_data):
        """Verify spectral_plot has axis labels."""
        _, ax = ts.spectral_plot(timeseries_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""


# --- autocorrelation_plot ---


class TestAutocorrelationPlot:
    """Tests for autocorrelation_plot."""

    def test_returns_figure_and_axes(self, timeseries_data):
        """Verify autocorrelation_plot returns Figure and Axes."""
        fig, ax = ts.autocorrelation_plot(timeseries_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, timeseries_data):
        """Verify autocorrelation_plot respects custom fig and ax."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = ts.autocorrelation_plot(
            timeseries_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_lines(self, timeseries_data):
        """Verify autocorrelation_plot creates line objects."""
        _, ax = ts.autocorrelation_plot(timeseries_data)
        assert len(ax.get_lines()) > 0

    def test_has_labels(self, timeseries_data):
        """Verify autocorrelation_plot has axis labels."""
        _, ax = ts.autocorrelation_plot(timeseries_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""


# --- complex_demodulation_amplitude_plot ---


class TestComplexDemodulationAmplitudePlot:
    """Tests for complex_demodulation_amplitude_plot."""

    def test_returns_figure_and_axes(self, timeseries_data):
        """Verify complex_demodulation_amplitude_plot returns Figure."""
        fig, ax = ts.complex_demodulation_amplitude_plot(timeseries_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_t(self, univariate_data):
        """Verify error raised when t is missing."""
        with pytest.raises(ValueError, match="requires t"):
            ts.complex_demodulation_amplitude_plot(univariate_data)

    def test_custom_fig_ax(self, timeseries_data):
        """Verify function respects custom fig and ax."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = ts.complex_demodulation_amplitude_plot(
            timeseries_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_has_labels(self, timeseries_data):
        """Verify plot has axis labels and title."""
        _, ax = ts.complex_demodulation_amplitude_plot(timeseries_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""


# --- complex_demodulation_phase_plot ---


class TestComplexDemodulationPhasePlot:
    """Tests for complex_demodulation_phase_plot."""

    def test_returns_figure_and_axes(self, timeseries_data):
        """Verify complex_demodulation_phase_plot returns Figure."""
        fig, ax = ts.complex_demodulation_phase_plot(timeseries_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_t(self, univariate_data):
        """Verify error raised when t is missing."""
        with pytest.raises(ValueError, match="requires t"):
            ts.complex_demodulation_phase_plot(univariate_data)

    def test_creates_two_lines(self, timeseries_data):
        """Verify phase plot creates multiple lines."""
        _, ax = ts.complex_demodulation_phase_plot(timeseries_data)
        assert len(ax.get_lines()) >= MIN_PHASE_PLOT_LINES

    def test_has_labels(self, timeseries_data):
        """Verify plot has axis labels and title."""
        _, ax = ts.complex_demodulation_phase_plot(timeseries_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""
