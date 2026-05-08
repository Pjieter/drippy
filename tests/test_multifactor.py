"""Tests for the drippy.multifactor module."""

from __future__ import annotations
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import drippy.multifactor as mf
from drippy.data import EDAData


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def multifactor_data():
    rng = np.random.default_rng(42)
    a = np.tile([-1.0, 1.0], 4)
    b = np.tile(np.repeat([-1.0, 1.0], 2), 2)
    c = np.repeat([-1.0, 1.0], 4)
    y = 2.0 * a + 1.5 * b - 0.5 * c + rng.normal(scale=0.1, size=8)
    return EDAData(y, factors={"A": a, "B": b, "C": c})


@pytest.fixture
def two_factor_data():
    rng = np.random.default_rng(42)
    a = np.tile([-1.0, 1.0], 4)
    b = np.repeat([-1.0, 1.0], 4)
    y = a + b + rng.normal(scale=0.1, size=8)
    return EDAData(y, factors={"A": a, "B": b})


@pytest.fixture
def no_factors_data():
    return EDAData(np.random.default_rng(42).normal(size=8))


@pytest.fixture
def empty_factors_data():
    return EDAData(np.random.default_rng(42).normal(size=8), factors={})


class TestDoeScatterPlot:
    def test_returns_figure_and_array(self, multifactor_data):
        fig, axes = mf.doe_scatter_plot(multifactor_data)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (3,)

    def test_requires_factors(self, no_factors_data):
        with pytest.raises(ValueError, match="requires factors"):
            mf.doe_scatter_plot(no_factors_data)

    def test_custom_fig_axes(self, multifactor_data):
        fig_in, axes_in = plt.subplots(1, 3)
        fig_out, axes_out = mf.doe_scatter_plot(
            multifactor_data, fig=fig_in, axes=axes_in
        )
        assert fig_out is fig_in
        assert axes_out is axes_in

    def test_one_subplot_per_factor(self, multifactor_data):
        _, axes = mf.doe_scatter_plot(multifactor_data)
        assert len(axes) == len(multifactor_data.factors)

    def test_rejects_empty_factors(self, empty_factors_data):
        with pytest.raises(ValueError, match="requires factors"):
            mf.doe_scatter_plot(empty_factors_data)

    def test_rejects_mismatched_fig_axes(self, multifactor_data):
        _, axes1 = plt.subplots(1, 3)
        fig2 = plt.figure()
        with pytest.raises(ValueError, match="do not belong"):
            mf.doe_scatter_plot(multifactor_data, fig=fig2, axes=axes1)

    def test_axes_shape_validation(self, multifactor_data):
        fig, wrong_axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match=r"axes must have shape"):
            mf.doe_scatter_plot(multifactor_data, fig=fig, axes=wrong_axes)


class TestDoeMeanPlot:
    def test_returns_figure_and_array(self, multifactor_data):
        fig, axes = mf.doe_mean_plot(multifactor_data)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (3,)

    def test_requires_factors(self, no_factors_data):
        with pytest.raises(ValueError, match="requires factors"):
            mf.doe_mean_plot(no_factors_data)

    def test_custom_fig_axes(self, multifactor_data):
        fig_in, axes_in = plt.subplots(1, 3)
        fig_out, axes_out = mf.doe_mean_plot(
            multifactor_data, fig=fig_in, axes=axes_in
        )
        assert fig_out is fig_in
        assert axes_out is axes_in

    def test_each_subplot_has_grand_mean_line(self, multifactor_data):
        _, axes = mf.doe_mean_plot(multifactor_data)
        for ax in axes:
            assert len(ax.get_lines()) >= 2

    def test_rejects_empty_factors(self, empty_factors_data):
        with pytest.raises(ValueError, match="requires factors"):
            mf.doe_mean_plot(empty_factors_data)

    def test_rejects_mismatched_fig_axes(self, multifactor_data):
        _, axes1 = plt.subplots(1, 3)
        fig2 = plt.figure()
        with pytest.raises(ValueError, match="do not belong"):
            mf.doe_mean_plot(multifactor_data, fig=fig2, axes=axes1)

    def test_axes_shape_validation(self, multifactor_data):
        fig, wrong_axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match=r"axes must have shape"):
            mf.doe_mean_plot(multifactor_data, fig=fig, axes=wrong_axes)


class TestDoeSdPlot:
    def test_returns_figure_and_array(self, multifactor_data):
        fig, axes = mf.doe_sd_plot(multifactor_data)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (3,)

    def test_requires_factors(self, no_factors_data):
        with pytest.raises(ValueError, match="requires factors"):
            mf.doe_sd_plot(no_factors_data)

    def test_custom_fig_axes(self, multifactor_data):
        fig_in, axes_in = plt.subplots(1, 3)
        fig_out, axes_out = mf.doe_sd_plot(
            multifactor_data, fig=fig_in, axes=axes_in
        )
        assert fig_out is fig_in
        assert axes_out is axes_in

    def test_each_subplot_has_overall_sd_line(self, multifactor_data):
        _, axes = mf.doe_sd_plot(multifactor_data)
        for ax in axes:
            assert len(ax.get_lines()) >= 2

    def test_rejects_empty_factors(self, empty_factors_data):
        with pytest.raises(ValueError, match="requires factors"):
            mf.doe_sd_plot(empty_factors_data)

    def test_rejects_mismatched_fig_axes(self, multifactor_data):
        _, axes1 = plt.subplots(1, 3)
        fig2 = plt.figure()
        with pytest.raises(ValueError, match="do not belong"):
            mf.doe_sd_plot(multifactor_data, fig=fig2, axes=axes1)

    def test_axes_shape_validation(self, multifactor_data):
        fig, wrong_axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match=r"axes must have shape"):
            mf.doe_sd_plot(multifactor_data, fig=fig, axes=wrong_axes)


class TestContourPlot:
    def test_returns_figure_and_axes(self, two_factor_data):
        fig, ax = mf.contour_plot(two_factor_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_factors(self, no_factors_data):
        with pytest.raises(ValueError, match="requires factors"):
            mf.contour_plot(no_factors_data)

    def test_rejects_empty_factors(self, empty_factors_data):
        with pytest.raises(ValueError, match="requires factors"):
            mf.contour_plot(empty_factors_data)

    def test_requires_exactly_two_factors(self, multifactor_data):
        with pytest.raises(ValueError, match="exactly 2 factors"):
            mf.contour_plot(multifactor_data)

    def test_custom_fig_ax(self, two_factor_data):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = mf.contour_plot(
            two_factor_data, fig=fig_in, ax=ax_in
        )
        assert fig_out is fig_in
        assert ax_out is ax_in

    def test_doe_variant_adds_scatter_markers(self, two_factor_data):
        _, ax_base = mf.contour_plot(two_factor_data, doe=False)
        n_base = len(ax_base.collections)
        plt.close("all")
        _, ax_doe = mf.contour_plot(two_factor_data, doe=True)
        assert len(ax_doe.collections) > n_base
