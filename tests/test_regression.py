"""Tests for drippy.regression (Phase 3 — regression model)."""

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import drippy.regression as reg
from drippy import EDAData

# --- Fixtures ---


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def regression_data():
    """50-point linear relationship with noise."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 50)
    return EDAData(rng.normal(loc=x), x=x)


@pytest.fixture
def no_x_data():
    """EDAData without x — used to test requires-x validation."""
    rng = np.random.default_rng(42)
    return EDAData(rng.normal(size=50))


# --- TestScatterPlot ---


class TestScatterPlot:
    def test_returns_figure_and_axes(self, regression_data):
        fig, ax = reg.scatter_plot(regression_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, no_x_data):
        with pytest.raises(ValueError, match="scatter_plot requires x"):
            reg.scatter_plot(no_x_data)

    def test_custom_fig_ax(self, regression_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = reg.scatter_plot(
            regression_data, fig=fig_in, ax=ax_in
        )
        assert fig_out is fig_in
        assert ax_out is ax_in


# --- TestSixPlot ---


class TestSixPlot:
    def test_returns_figure_and_axes_array(self, regression_data):
        fig, axes = reg.six_plot(regression_data)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 3)

    def test_axes_are_all_axes(self, regression_data):
        _, axes = reg.six_plot(regression_data)
        for ax in axes.flat:
            assert isinstance(ax, Axes)

    def test_requires_x(self, no_x_data):
        with pytest.raises(ValueError, match="six_plot requires x"):
            reg.six_plot(no_x_data)

    def test_custom_axes(self, regression_data):
        fig_in, axes_in = plt.subplots(2, 3)
        fig_out, axes_out = reg.six_plot(
            regression_data, fig=fig_in, axes=axes_in
        )
        assert fig_out is fig_in
        assert axes_out is axes_in

    def test_wrong_axes_shape_raises(self, regression_data):
        _, axes = plt.subplots(1, 6)
        with pytest.raises(ValueError, match="axes must have shape"):
            reg.six_plot(regression_data, axes=axes)

    def test_all_panels_have_titles(self, regression_data):
        _, axes = reg.six_plot(regression_data)
        for ax in axes.flat:
            assert ax.get_title() != ""


# --- TestLinearCorrelationPlot ---


class TestLinearCorrelationPlot:
    def test_returns_figure_and_axes(self, regression_data):
        fig, ax = reg.linear_correlation_plot(regression_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, no_x_data):
        with pytest.raises(
            ValueError, match="linear_correlation_plot requires x"
        ):
            reg.linear_correlation_plot(no_x_data)

    def test_custom_window(self, regression_data):
        fig, ax = reg.linear_correlation_plot(regression_data, window=5)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, regression_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = reg.linear_correlation_plot(
            regression_data, fig=fig_in, ax=ax_in
        )
        assert fig_out is fig_in
        assert ax_out is ax_in

    def test_line_has_data(self, regression_data):
        _, ax = reg.linear_correlation_plot(regression_data)
        lines = ax.get_lines()
        assert len(lines) > 0


# --- TestLinearInterceptPlot ---


class TestLinearInterceptPlot:
    def test_returns_figure_and_axes(self, regression_data):
        fig, ax = reg.linear_intercept_plot(regression_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, no_x_data):
        with pytest.raises(
            ValueError, match="linear_intercept_plot requires x"
        ):
            reg.linear_intercept_plot(no_x_data)

    def test_custom_window(self, regression_data):
        fig, ax = reg.linear_intercept_plot(regression_data, window=5)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, regression_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = reg.linear_intercept_plot(
            regression_data, fig=fig_in, ax=ax_in
        )
        assert fig_out is fig_in
        assert ax_out is ax_in

    def test_reference_line_present(self, regression_data):
        _, ax = reg.linear_intercept_plot(regression_data)
        # reference line is an axhline — check at least 2 lines (data + hline)
        assert len(ax.get_lines()) >= 2


# --- TestLinearSlopePlot ---


class TestLinearSlopePlot:
    def test_returns_figure_and_axes(self, regression_data):
        fig, ax = reg.linear_slope_plot(regression_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, no_x_data):
        with pytest.raises(ValueError, match="linear_slope_plot requires x"):
            reg.linear_slope_plot(no_x_data)

    def test_custom_window(self, regression_data):
        fig, ax = reg.linear_slope_plot(regression_data, window=5)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, regression_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = reg.linear_slope_plot(
            regression_data, fig=fig_in, ax=ax_in
        )
        assert fig_out is fig_in
        assert ax_out is ax_in

    def test_reference_line_present(self, regression_data):
        _, ax = reg.linear_slope_plot(regression_data)
        assert len(ax.get_lines()) >= 2


# --- TestLinearResidualSdPlot ---


class TestLinearResidualSdPlot:
    def test_returns_figure_and_axes(self, regression_data):
        fig, ax = reg.linear_residual_sd_plot(regression_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, no_x_data):
        with pytest.raises(
            ValueError, match="linear_residual_sd_plot requires x"
        ):
            reg.linear_residual_sd_plot(no_x_data)

    def test_custom_window(self, regression_data):
        fig, ax = reg.linear_residual_sd_plot(regression_data, window=5)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, regression_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = reg.linear_residual_sd_plot(
            regression_data, fig=fig_in, ax=ax_in
        )
        assert fig_out is fig_in
        assert ax_out is ax_in

    def test_reference_line_present(self, regression_data):
        _, ax = reg.linear_residual_sd_plot(regression_data)
        assert len(ax.get_lines()) >= 2


# --- TestFluentMethods ---


class TestFluentMethods:
    """Verify EDAData fluent delegation for Phase 3 methods."""

    def test_six_plot_fluent(self, regression_data):
        fig, axes = regression_data.six_plot()
        assert isinstance(fig, Figure)
        assert axes.shape == (2, 3)

    def test_linear_correlation_plot_fluent(self, regression_data):
        fig, ax = regression_data.linear_correlation_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_linear_intercept_plot_fluent(self, regression_data):
        fig, ax = regression_data.linear_intercept_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_linear_slope_plot_fluent(self, regression_data):
        fig, ax = regression_data.linear_slope_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_linear_residual_sd_plot_fluent(self, regression_data):
        fig, ax = regression_data.linear_residual_sd_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
