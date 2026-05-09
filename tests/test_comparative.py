"""Tests for drippy.comparative (Phase 4)."""

from __future__ import annotations
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import drippy.comparative as cp
from drippy.data import EDAData


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def block_data():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(24)
    treatment = np.tile([1, 2, 3, 4], 6)
    block = np.repeat([1, 2, 3, 4, 5, 6], 4)
    return EDAData(y=y, factors={"treatment": treatment, "block": block})


@pytest.fixture
def youden_data():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(20)
    x = y + rng.normal(scale=0.1, size=20)
    return EDAData(y=y, x=x)


@pytest.fixture
def star_data():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(5)
    return EDAData(
        y=y,
        factors={
            "A": rng.standard_normal(5),
            "B": rng.standard_normal(5),
            "C": rng.standard_normal(5),
        },
    )


class TestBlockPlot:
    def test_returns_figure_and_axes(self, block_data):
        fig, ax = cp.block_plot(block_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_factors_not_none(self):
        data = EDAData(y=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="treatment"):
            cp.block_plot(data)

    def test_requires_treatment_key(self):
        factors = {"block": np.array([1, 1, 2])}
        data = EDAData(y=[1.0, 2.0, 3.0], factors=factors)
        with pytest.raises(ValueError, match="treatment"):
            cp.block_plot(data)

    def test_requires_block_key(self):
        factors = {"treatment": np.array([1, 2, 3])}
        data = EDAData(y=[1.0, 2.0, 3.0], factors=factors)
        with pytest.raises(ValueError, match="block"):
            cp.block_plot(data)

    def test_custom_fig_ax(self, block_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = cp.block_plot(block_data, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in

    def test_has_labels(self, block_data):
        _, ax = cp.block_plot(block_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""

    def test_legend_present(self, block_data):
        _, ax = cp.block_plot(block_data)
        assert ax.get_legend() is not None

    def test_one_line_per_block(self, block_data):
        _, ax = cp.block_plot(block_data)
        n_blocks = len(np.unique(block_data.factors["block"]))
        assert len(ax.get_lines()) >= n_blocks


class TestYoudenPlot:
    def test_returns_figure_and_axes(self, youden_data):
        fig, ax = cp.youden_plot(youden_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_x(self):
        data = EDAData(y=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="requires x"):
            cp.youden_plot(data)

    def test_custom_fig_ax(self, youden_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = cp.youden_plot(youden_data, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in

    def test_has_labels(self, youden_data):
        _, ax = cp.youden_plot(youden_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""

    def test_equality_line_present(self, youden_data):
        _, ax = cp.youden_plot(youden_data)
        assert len(ax.get_lines()) >= 1

    def test_median_lines_present(self, youden_data):
        _, ax = cp.youden_plot(youden_data)
        assert len(ax.get_lines()) >= 3

    def test_doe_adds_scatter_collection(self, youden_data):
        _, ax_no_doe = cp.youden_plot(youden_data)
        n_base = len(ax_no_doe.collections)
        plt.close("all")
        _, ax_doe = cp.youden_plot(youden_data, doe=True)
        assert len(ax_doe.collections) == n_base + 1

    def test_legend_present(self, youden_data):
        _, ax = cp.youden_plot(youden_data)
        assert ax.get_legend() is not None


class TestStarPlot:
    def test_returns_figure_and_axes(self, star_data):
        fig, ax = cp.star_plot(star_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_factors(self):
        data = EDAData(y=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="requires factors"):
            cp.star_plot(data)

    def test_custom_fig_ax(self, star_data):
        fig_in, ax_in = plt.subplots(subplot_kw={"projection": "polar"})
        fig_out, ax_out = cp.star_plot(star_data, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in

    def test_has_title(self, star_data):
        _, ax = cp.star_plot(star_data)
        assert ax.get_title() != ""

    def test_one_polygon_per_observation(self, star_data):
        _, ax = cp.star_plot(star_data)
        assert len(ax.get_lines()) == len(star_data.y)

    def test_spoke_labels_match_variables(self, star_data):
        _, ax = cp.star_plot(star_data)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "y" in labels
        assert "A" in labels
        assert "B" in labels
        assert "C" in labels

    def test_single_observation(self):
        data = EDAData(y=[3.0], factors={"A": np.array([1.0])})
        fig, ax = cp.star_plot(data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestFluentMethods:
    def test_block_plot_fluent(self, block_data):
        fig, ax = block_data.block_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_youden_plot_fluent(self, youden_data):
        fig, ax = youden_data.youden_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_star_plot_fluent(self, star_data):
        fig, ax = star_data.star_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
