"""Tests for the drippy.data module."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mpl.use("Agg")

from drippy.data import EDAData


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


# --- Fixtures ---


@pytest.fixture
def univariate_data():
    """EDAData with univariate (y-only) data."""
    return EDAData(np.random.default_rng(42).normal(size=50))


@pytest.fixture
def timeseries_data():
    """EDAData with time series (y and t) data."""
    rng = np.random.default_rng(42)
    return EDAData(rng.normal(size=100), t=np.linspace(0, 10, 100))


@pytest.fixture
def onefactor_data():
    """EDAData with one categorical factor."""
    rng = np.random.default_rng(42)
    return EDAData(
        rng.normal(size=50),
        x=np.repeat(["A", "B", "C", "D", "E"], 10),
    )


@pytest.fixture
def multifactor_data():
    """EDAData with multiple factors (DOE)."""
    rng = np.random.default_rng(42)
    return EDAData(
        rng.normal(size=16),
        factors={"A": np.tile([-1, 1], 8), "B": np.repeat([-1, 1], 8)},
    )


@pytest.fixture
def regression_data():
    """EDAData with continuous predictor (regression)."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 50)
    return EDAData(rng.normal(loc=x), x=x)


# --- Construction ---


class TestEDADataConstruction:
    """Tests for EDAData construction."""

    def test_y_only(self):
        """EDAData with only y should have None for x, t, factors."""
        data = EDAData([1.0, 2.0, 3.0])
        assert isinstance(data.y, np.ndarray)
        assert data.x is None
        assert data.t is None
        assert data.factors is None

    def test_with_x(self):
        """X should be stored as ndarray."""
        data = EDAData([1.0, 2.0, 3.0], x=[4.0, 5.0, 6.0])
        assert isinstance(data.x, np.ndarray)
        np.testing.assert_array_equal(data.x, [4.0, 5.0, 6.0])

    def test_with_t(self):
        """T should be stored as float ndarray."""
        data = EDAData([1.0, 2.0, 3.0], t=[0.1, 0.2, 0.3])
        assert isinstance(data.t, np.ndarray)
        assert data.t.dtype == float

    def test_with_factors(self):
        """Factors should be stored as dict of ndarrays."""
        data = EDAData([1.0, 2.0, 3.0], factors={"A": [1, 2, 3]})
        assert isinstance(data.factors, dict)
        assert isinstance(data.factors["A"], np.ndarray)

    def test_list_inputs_stored_as_ndarray(self):
        """List inputs should be converted to ndarray."""
        data = EDAData([1, 2, 3])
        assert isinstance(data.y, np.ndarray)

    def test_t_stored_as_float(self):
        """T should be converted to float dtype."""
        data = EDAData([1, 2, 3], t=[1, 2, 3])
        assert data.t.dtype == float


# --- Validation ---


class TestEDADataValidation:
    """Tests for EDAData validation."""

    def test_empty_y_raises(self):
        """Empty y should raise ValueError."""
        with pytest.raises(ValueError, match="y cannot be empty"):
            EDAData([])

    def test_multidim_y_raises(self):
        """Multi-dimensional y should raise ValueError."""
        with pytest.raises(ValueError, match="y must be 1-dimensional"):
            EDAData(np.ones((3, 3)))

    def test_x_length_mismatch_raises(self):
        """X length mismatch should raise ValueError."""
        with pytest.raises(
            ValueError, match="x and y must have the same length"
        ):
            EDAData([1, 2, 3], x=[1, 2])

    def test_t_length_mismatch_raises(self):
        """T length mismatch should raise ValueError."""
        with pytest.raises(
            ValueError, match="t and y must have the same length"
        ):
            EDAData([1, 2, 3], t=[1, 2])

    def test_factors_length_mismatch_raises(self):
        """Factors length mismatch should raise ValueError."""
        with pytest.raises(
            ValueError, match="factors\\['A'\\] and y must have"
        ):
            EDAData([1, 2, 3], factors={"A": [1, 2]})

    def test_multidim_x_raises(self):
        """Multi-dimensional x should raise ValueError."""
        with pytest.raises(ValueError, match="x must be 1-dimensional"):
            EDAData([1, 2, 3, 4], x=np.ones((2, 2)))

    def test_multidim_t_raises(self):
        """Multi-dimensional t should raise ValueError."""
        with pytest.raises(ValueError, match="t must be 1-dimensional"):
            EDAData([1, 2, 3, 4], t=np.ones((2, 2)))

    def test_multidim_factor_raises(self):
        """Multi-dimensional factor should raise ValueError."""
        with pytest.raises(
            ValueError, match="factors\\['A'\\] must be 1-dimensional"
        ):
            EDAData([1, 2, 3, 4], factors={"A": np.ones((2, 2))})


# --- Fluent methods ---


class TestFluentMethods:
    """Each method must delegate to the correct module function."""

    def test_run_sequence_plot(self, univariate_data):
        fig, ax = univariate_data.run_sequence_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_histogram(self, univariate_data):
        fig, ax = univariate_data.histogram()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_lag_plot(self, univariate_data):
        fig, ax = univariate_data.lag_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_normal_probability_plot(self, univariate_data):
        fig, ax, _ = univariate_data.normal_probability_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_four_plot(self, univariate_data):
        fig, axes = univariate_data.four_plot()
        assert isinstance(fig, Figure)
        assert len(axes) == 4

    def test_spectral_plot(self, timeseries_data):
        fig, ax = timeseries_data.spectral_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_autocorrelation_plot(self, timeseries_data):
        fig, ax = timeseries_data.autocorrelation_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_complex_demodulation_amplitude_plot(self, timeseries_data):
        fig, ax = timeseries_data.complex_demodulation_amplitude_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_complex_demodulation_phase_plot(self, timeseries_data):
        fig, ax = timeseries_data.complex_demodulation_phase_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_scatter_plot(self, onefactor_data):
        fig, ax = onefactor_data.scatter_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_box_plot(self, onefactor_data):
        fig, ax = onefactor_data.box_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_mean_plot(self, onefactor_data):
        fig, ax = onefactor_data.mean_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_sd_plot(self, onefactor_data):
        fig, ax = onefactor_data.sd_plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_kwargs_forwarded(self, univariate_data):
        """Kwargs must be forwarded to the underlying function."""
        _, ax = univariate_data.histogram(bins=5)
        assert len(ax.patches) == 5

    def test_doe_scatter_plot(self, multifactor_data):
        fig, axes = multifactor_data.doe_scatter_plot()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
