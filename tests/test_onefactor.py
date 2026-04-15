"""Tests for the drippy.onefactor module."""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mpl.use("Agg")

import drippy.onefactor as of
from drippy.data import EDAData

# --- Fixtures ---


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def univariate_data():
    """No x — for testing 'requires x' error paths."""
    return EDAData(np.random.default_rng(42).normal(size=50))


@pytest.fixture
def onefactor_data():
    """5 factor levels, 10 observations each."""
    rng = np.random.default_rng(42)
    y = rng.normal(loc=[0, 1, 2, 3, 4], scale=1, size=(10, 5)).flatten("F")
    x = np.repeat(["A", "B", "C", "D", "E"], 10)
    return EDAData(y, x=x)


@pytest.fixture
def bigroup_data():
    """2 factor levels — required by bihistogram and qq_plot."""
    rng = np.random.default_rng(42)
    y = np.concatenate([rng.normal(0, 1, 30), rng.normal(2, 1, 30)])
    x = np.repeat(["Control", "Treatment"], 30)
    return EDAData(y, x=x)


# --- scatter_plot ---


class TestScatterPlot:
    def test_returns_figure_and_axes(self, onefactor_data):
        fig, ax = of.scatter_plot(onefactor_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, univariate_data):
        with pytest.raises(ValueError, match="requires x"):
            of.scatter_plot(univariate_data)

    def test_custom_fig_ax(self, onefactor_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = of.scatter_plot(
            onefactor_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_has_labels(self, onefactor_data):
        _, ax = of.scatter_plot(onefactor_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""


# --- box_plot ---


class TestBoxPlot:
    def test_returns_figure_and_axes(self, onefactor_data):
        fig, ax = of.box_plot(onefactor_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, univariate_data):
        with pytest.raises(ValueError, match="requires x"):
            of.box_plot(univariate_data)

    def test_custom_fig_ax(self, onefactor_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = of.box_plot(onefactor_data, fig=provided_fig, ax=provided_ax)
        assert fig is provided_fig
        assert ax is provided_ax

    def test_one_box_per_level(self, onefactor_data):
        _, ax = of.box_plot(onefactor_data)
        n_levels = len(np.unique(onefactor_data.x))
        # Verify at least one line per level (median line minimum)
        assert len(ax.get_lines()) >= n_levels

    def test_has_labels(self, onefactor_data):
        _, ax = of.box_plot(onefactor_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""


# --- bihistogram ---


class TestBihistogram:
    def test_returns_figure_and_axes(self, bigroup_data):
        fig, axes = of.bihistogram(bigroup_data)
        assert isinstance(fig, Figure)
        assert len(axes) == 2

    def test_requires_x(self, univariate_data):
        with pytest.raises(ValueError, match="requires x"):
            of.bihistogram(univariate_data)

    def test_requires_exactly_two_levels(self, onefactor_data):
        with pytest.raises(ValueError, match="exactly 2 factor levels"):
            of.bihistogram(onefactor_data)

    def test_both_subplots_have_patches(self, bigroup_data):
        _, axes = of.bihistogram(bigroup_data)
        for ax in axes:
            assert len(ax.patches) > 0

    def test_subplot_titles_contain_level_names(self, bigroup_data):
        levels = np.unique(bigroup_data.x)
        _, axes = of.bihistogram(bigroup_data)
        assert levels[0] in axes[0].get_title()
        assert levels[1] in axes[1].get_title()


# --- qq_plot ---


class TestQqPlot:
    def test_returns_figure_and_axes(self, bigroup_data):
        fig, ax = of.qq_plot(bigroup_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, univariate_data):
        with pytest.raises(ValueError, match="requires x"):
            of.qq_plot(univariate_data)

    def test_requires_exactly_two_levels(self, onefactor_data):
        with pytest.raises(ValueError, match="exactly 2 factor levels"):
            of.qq_plot(onefactor_data)

    def test_has_reference_line(self, bigroup_data):
        _, ax = of.qq_plot(bigroup_data)
        # scatter + reference line
        assert len(ax.get_lines()) >= 1

    def test_has_labels(self, bigroup_data):
        _, ax = of.qq_plot(bigroup_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""


# --- mean_plot ---


class TestMeanPlot:
    def test_returns_figure_and_axes(self, onefactor_data):
        fig, ax = of.mean_plot(onefactor_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, univariate_data):
        with pytest.raises(ValueError, match="requires x"):
            of.mean_plot(univariate_data)

    def test_has_grand_mean_line(self, onefactor_data):
        _, ax = of.mean_plot(onefactor_data)
        # line for group means + horizontal grand mean
        assert len(ax.get_lines()) >= 2

    def test_has_labels(self, onefactor_data):
        _, ax = of.mean_plot(onefactor_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""


# --- sd_plot ---


class TestSdPlot:
    def test_returns_figure_and_axes(self, onefactor_data):
        fig, ax = of.sd_plot(onefactor_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self, univariate_data):
        with pytest.raises(ValueError, match="requires x"):
            of.sd_plot(univariate_data)

    def test_has_overall_sd_line(self, onefactor_data):
        _, ax = of.sd_plot(onefactor_data)
        assert len(ax.get_lines()) >= 2

    def test_has_labels(self, onefactor_data):
        _, ax = of.sd_plot(onefactor_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""
