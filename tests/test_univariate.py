"""Tests for the drippy.univariate module."""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mpl.use("Agg")

import drippy.univariate as uv
from drippy.data import EDAData

# --- Fixtures ---


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def univariate_data():
    rng = np.random.default_rng(42)
    return EDAData(5 + rng.normal(size=100))


@pytest.fixture
def univariate_data_with_t():
    rng = np.random.default_rng(42)
    return EDAData(5 + rng.normal(size=100), t=np.linspace(1, 10, 100))


@pytest.fixture
def positive_data():
    rng = np.random.default_rng(42)
    return EDAData(np.abs(5 + rng.normal(size=50)) + 1)


@pytest.fixture
def positive_data_with_x():
    """Positive y with positive x — for box_cox_linearity_plot."""
    rng = np.random.default_rng(42)
    x = np.linspace(0.1, 5.0, 50)
    y = np.abs(2 * x + rng.normal(size=50)) + 0.5
    return EDAData(y, x=x)


# --- run_sequence_plot ---


class TestRunSequencePlot:
    def test_returns_figure_and_axes(self, univariate_data):
        fig, ax = uv.run_sequence_plot(univariate_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_uses_index_when_no_t(self, univariate_data):
        _, ax = uv.run_sequence_plot(univariate_data)
        lines = ax.get_lines()
        assert len(lines) >= 1
        x_data, _ = lines[0].get_data()
        expected = np.arange(len(univariate_data.y))
        np.testing.assert_array_equal(x_data, expected)

    def test_uses_t_when_provided(self, univariate_data_with_t):
        _, ax = uv.run_sequence_plot(univariate_data_with_t)
        lines = ax.get_lines()
        x_data, _ = lines[0].get_data()
        np.testing.assert_array_equal(x_data, univariate_data_with_t.t)

    def test_custom_fig_ax(self, univariate_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = uv.run_sequence_plot(
            univariate_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_has_labels(self, univariate_data):
        _, ax = uv.run_sequence_plot(univariate_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""


# --- lag_plot ---


class TestLagPlot:
    def test_returns_figure_and_axes(self, univariate_data):
        fig, ax = uv.lag_plot(univariate_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, univariate_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = uv.lag_plot(
            univariate_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_data_length(self, univariate_data):
        lag = 3
        _, ax = uv.lag_plot(univariate_data, lag=lag)
        scatter_data = ax.collections[0].get_offsets()
        assert len(scatter_data) == len(univariate_data.y) - lag

    def test_data_orientation(self, univariate_data):
        lag = 2
        _, ax = uv.lag_plot(univariate_data, lag=lag)
        scatter_data = ax.collections[0].get_offsets()
        np.testing.assert_array_equal(
            scatter_data[:, 0], univariate_data.y[:-lag]
        )
        np.testing.assert_array_equal(
            scatter_data[:, 1], univariate_data.y[lag:]
        )

    def test_negative_lag_raises(self, univariate_data):
        with pytest.raises(ValueError, match="Lag must be a positive integer"):
            uv.lag_plot(univariate_data, lag=-1)

    def test_zero_lag_raises(self, univariate_data):
        with pytest.raises(ValueError, match="Lag must be a positive integer"):
            uv.lag_plot(univariate_data, lag=0)

    def test_lag_too_large_raises(self, univariate_data):
        with pytest.raises(
            ValueError, match="Lag must be less than the length"
        ):
            uv.lag_plot(univariate_data, lag=len(univariate_data.y))

    def test_float_lag_converted(self, univariate_data):
        fig, _ = uv.lag_plot(univariate_data, lag=2.7)
        assert isinstance(fig, Figure)


# --- histogram ---


class TestHistogram:
    def test_returns_figure_and_axes(self, univariate_data):
        fig, ax = uv.histogram(univariate_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, univariate_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = uv.histogram(
            univariate_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_patches(self, univariate_data):
        _, ax = uv.histogram(univariate_data)
        assert len(ax.patches) > 0

    def test_integer_bins(self, univariate_data):
        _, ax = uv.histogram(univariate_data, bins=10)
        assert len(ax.patches) == 10

    def test_negative_bins_raises(self, univariate_data):
        with pytest.raises(ValueError):
            uv.histogram(univariate_data, bins=-5)

    def test_zero_bins_raises(self, univariate_data):
        with pytest.raises(ValueError):
            uv.histogram(univariate_data, bins=0)


# --- normal_probability_plot ---


class TestNormalProbabilityPlot:
    def test_returns_figure_and_axes(self, univariate_data):
        fig, ax = uv.normal_probability_plot(univariate_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, univariate_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = uv.normal_probability_plot(
            univariate_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_returns_rsquared_when_requested(self, univariate_data):
        result = uv.normal_probability_plot(
            univariate_data, return_rsquared=True
        )
        assert len(result) == 3
        _, _, rsq = result
        assert isinstance(rsq, float)
        assert 0 <= rsq <= 1

    def test_returns_two_by_default(self, univariate_data):
        result = uv.normal_probability_plot(univariate_data)
        assert len(result) == 2


# --- four_plot ---


class TestFourPlot:
    def test_returns_figure_and_axes(self, univariate_data):
        fig, axes = uv.four_plot(univariate_data)
        assert isinstance(fig, Figure)
        assert len(axes) == 4

    def test_all_subplots_have_content(self, univariate_data):
        _, axes = uv.four_plot(univariate_data)
        for ax in axes:
            has_content = (
                len(ax.get_lines()) > 0
                or len(ax.patches) > 0
                or len(ax.collections) > 0
            )
            assert has_content

    def test_wrong_axes_shape_raises(self, univariate_data):
        fig, axes = plt.subplots(1, 3)
        with pytest.raises(
            ValueError, match=r"Axes must be an iterable of \(2, 2\)"
        ):
            uv.four_plot(univariate_data, fig=fig, axes=axes)

    def test_custom_fig_axes(self, univariate_data):
        provided_fig, provided_axes = plt.subplots(2, 2)
        fig, _ = uv.four_plot(
            univariate_data, fig=provided_fig, axes=provided_axes
        )
        assert fig is provided_fig


# --- ppcc_plot ---


class TestPpccPlot:
    def test_returns_figure_and_two_axes(self, univariate_data):
        fig, axes = uv.ppcc_plot(univariate_data)
        assert isinstance(fig, Figure)
        assert len(axes) == 2

    def test_subplot_titles(self, univariate_data):
        _, axes = uv.ppcc_plot(univariate_data)
        assert axes[0].get_title() == "Rough PPCC Plot"
        assert axes[1].get_title() == "Fine PPCC Plot"

    def test_invalid_rough_range_raises(self, univariate_data):
        with pytest.raises(
            ValueError, match="Rough range must contain exactly 2"
        ):
            uv.ppcc_plot(univariate_data, rough_range=(0.1, 0.5, 0.9))

    def test_inverted_range_raises(self, univariate_data):
        with pytest.raises(ValueError, match="min < max"):
            uv.ppcc_plot(univariate_data, rough_range=(2, -2))

    def test_negative_n_raises(self, univariate_data):
        with pytest.raises(
            ValueError, match="Number of points must be positive"
        ):
            uv.ppcc_plot(univariate_data, n_rough=-1)


# --- weibull_plot ---


class TestWeibullPlot:
    def test_returns_figure_and_axes(self, positive_data):
        fig, ax = uv.weibull_plot(positive_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_non_positive_raises(self):
        data = EDAData(np.array([1.0, -2.0, 3.0]))
        with pytest.raises(ValueError, match="all y values to be positive"):
            uv.weibull_plot(data)

    def test_zero_raises(self):
        data = EDAData(np.array([1.0, 0.0, 3.0]))
        with pytest.raises(ValueError, match="all y values to be positive"):
            uv.weibull_plot(data)


# --- probability_plot ---


class TestProbabilityPlot:
    def test_returns_figure_and_axes(self, univariate_data):
        fig, ax = uv.probability_plot(univariate_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_default_distribution_norm(self, univariate_data):
        _, ax = uv.probability_plot(univariate_data)
        assert "Norm" in ax.get_title()

    def test_custom_distribution_in_title(self, univariate_data):
        _, ax = uv.probability_plot(univariate_data, distribution="expon")
        assert "Expon" in ax.get_title()

    def test_invalid_distribution_raises(self, univariate_data):
        with pytest.raises(ValueError):
            uv.probability_plot(univariate_data, distribution="invalid_dist")


# --- box_cox_normality_plot ---


class TestBoxCoxNormalityPlot:
    def test_returns_figure_and_axes(self, positive_data):
        fig, axes = uv.box_cox_normality_plot(positive_data)
        assert isinstance(fig, Figure)
        assert axes.shape == (2, 2)

    def test_non_positive_raises(self):
        data = EDAData(np.array([1.0, -2.0, 3.0]))
        with pytest.raises(ValueError, match="all y values to be positive"):
            uv.box_cox_normality_plot(data)

    def test_wrong_axes_shape_raises(self, positive_data):
        fig, axes = plt.subplots(1, 3)
        with pytest.raises(
            ValueError, match=r"Axes must be an iterable of \(2, 2\)"
        ):
            uv.box_cox_normality_plot(positive_data, fig=fig, axes=axes)

    def test_all_subplots_have_content(self, positive_data):
        _, axes = uv.box_cox_normality_plot(positive_data)
        for ax in axes.flatten():
            assert len(ax.get_lines()) > 0 or len(ax.patches) > 0


# --- bootstrap_plot ---


class TestBootstrapPlot:
    def test_returns_figure_and_axes(self, univariate_data):
        fig, ax = uv.bootstrap_plot(univariate_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_default_statistic_mean(self, univariate_data):
        _, ax = uv.bootstrap_plot(univariate_data)
        assert "mean" in ax.get_xlabel()

    def test_non_callable_statistic_raises(self, univariate_data):
        with pytest.raises(TypeError, match="Statistic must be callable"):
            uv.bootstrap_plot(univariate_data, statistic="mean")

    def test_negative_n_bootstrap_raises(self, univariate_data):
        with pytest.raises(ValueError, match="must be positive"):
            uv.bootstrap_plot(univariate_data, n_bootstrap=-1)


# --- box_cox_linearity_plot ---


class TestBoxCoxLinearityPlot:
    def test_returns_figure_and_axes(self, positive_data_with_x):
        fig, ax = uv.box_cox_linearity_plot(positive_data_with_x)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, univariate_data):
        with pytest.raises(ValueError, match="requires x"):
            uv.box_cox_linearity_plot(univariate_data)

    def test_requires_positive_x(self):
        data = EDAData([1.0, 2.0, 3.0], x=[-1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="all x values to be positive"):
            uv.box_cox_linearity_plot(data)

    def test_has_labels(self, positive_data_with_x):
        _, ax = uv.box_cox_linearity_plot(positive_data_with_x)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""
