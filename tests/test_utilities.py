"""Tests for the drippy.utilities module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from drippy.utilities import bl_filt
from drippy.utilities import get_figure_and_axes

matplotlib.use("Agg")  # Use non-interactive backend for testing


class TestGetFigureAndAxes:
    """Tests for get_figure_and_axes function."""

    def test_creates_new_figure_and_axes_when_none_provided(self):
        """Test that new Figure and Axes are created when none are provided."""
        fig, ax = get_figure_and_axes()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_uses_provided_figure(self):
        """Test that the provided Figure is used."""
        provided_fig = plt.figure()
        fig, ax = get_figure_and_axes(fig=provided_fig)

        assert fig is provided_fig
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_uses_provided_axes(self):
        """Test that the provided Axes is used."""
        provided_fig, provided_ax = plt.subplots()
        fig, ax = get_figure_and_axes(fig=provided_fig, ax=provided_ax)

        assert fig is provided_fig
        assert ax is provided_ax
        plt.close(fig)

    def test_creates_axes_on_provided_figure(self):
        """Test that Axes is created on the provided Figure when ax is None."""
        provided_fig = plt.figure()
        fig, ax = get_figure_and_axes(fig=provided_fig, ax=None)

        assert fig is provided_fig
        assert isinstance(ax, Axes)
        assert ax.figure is provided_fig
        plt.close(fig)

    def test_returns_tuple(self):
        """Test that the function returns a tuple."""
        result = get_figure_and_axes()

        assert isinstance(result, tuple)
        assert len(result) == 2
        plt.close(result[0])


class TestBlFilt:
    """Tests for bl_filt (Blackman filter) function."""

    def test_returns_array_with_correct_shape(self):
        """Test that output shape matches input shape."""
        y = np.random.randn(50)
        filtered = bl_filt(y, half_width=3)

        assert filtered.shape == y.shape
        assert isinstance(filtered, np.ndarray)

    def test_returns_finite_values(self):
        """Test that output contains only finite values."""
        y = np.random.randn(100)
        filtered = bl_filt(y, half_width=5)

        assert np.all(np.isfinite(filtered))

    def test_handles_different_half_widths(self):
        """Test that function works with different half_width values."""
        y = np.random.randn(50)

        # Test with minimal half_width
        filtered_small = bl_filt(y, half_width=1)
        assert filtered_small.shape == y.shape

        # Test with larger half_width
        filtered_large = bl_filt(y, half_width=10)
        assert filtered_large.shape == y.shape

    def test_handles_small_arrays(self):
        """Test that filter handles small arrays without errors."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        filtered = bl_filt(y, half_width=1)

        assert filtered.shape == y.shape
        assert np.all(np.isfinite(filtered))

    def test_handles_negative_values(self):
        """Test that filter works with negative values."""
        y = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        filtered = bl_filt(y, half_width=1)

        assert filtered.shape == y.shape
        assert np.all(np.isfinite(filtered))

    def test_handles_constant_signal(self):
        """Test that filter handles constant signals without errors."""
        y = np.ones(100) * 5.0
        filtered = bl_filt(y, half_width=5)

        assert filtered.shape == y.shape
        assert np.all(np.isfinite(filtered))
