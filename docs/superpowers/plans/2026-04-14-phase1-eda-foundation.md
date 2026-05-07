# Phase 1: EDA Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate drippy from a class-based API to a functional API with a single `EDAData` data container, refactor existing `univariate` and `timeseries` modules to functions, and implement `onefactor` as the first new plotting module.

**Architecture:** `EDAData` (in `data.py`) is a validated data bag exposing all plot functions as fluent methods via lazy imports. Plot functions live in model-specific modules (`univariate.py`, `timeseries.py`, `onefactor.py`) and accept `EDAData` as their first argument. `run_sequence_plot` is defined once in `univariate.py` and imported by `timeseries.py`. `scatter_plot` is defined in `onefactor.py` and will be re-imported by `regression.py` in Phase 3.

**Tech Stack:** Python 3.11+, matplotlib, numpy, scipy, astropy, lmfit, pytest, uv

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/drippy/data.py` | `EDAData` container + fluent methods |
| Rewrite | `src/drippy/univariate.py` | 11 standalone plot functions |
| Rewrite | `src/drippy/timeseries.py` | 5 standalone plot functions |
| Create | `src/drippy/onefactor.py` | 6 standalone plot functions |
| Rewrite | `src/drippy/__init__.py` | Flat re-exports of all public symbols |
| Create | `tests/test_data.py` | `EDAData` validation + fluent method tests |
| Rewrite | `tests/test_univariate.py` | Functional API tests |
| Rewrite | `tests/test_timeseries.py` | Functional API tests |
| Create | `tests/test_onefactor.py` | 1-factor function tests |

`src/drippy/utilities.py` is unchanged.

---

## Task 1: EDAData container

**Files:**
- Create: `src/drippy/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1.1: Write failing tests for EDAData**

Create `tests/test_data.py`:

```python
"""Tests for the drippy.data module."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

mpl.use("Agg")

from drippy.data import EDAData  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --- Fixtures ---

@pytest.fixture
def univariate_data():
    return EDAData(np.random.default_rng(42).normal(size=50))


@pytest.fixture
def timeseries_data():
    rng = np.random.default_rng(42)
    return EDAData(rng.normal(size=100), t=np.linspace(0, 10, 100))


@pytest.fixture
def onefactor_data():
    rng = np.random.default_rng(42)
    return EDAData(
        rng.normal(size=50),
        x=np.repeat(["A", "B", "C", "D", "E"], 10),
    )


@pytest.fixture
def multifactor_data():
    rng = np.random.default_rng(42)
    return EDAData(
        rng.normal(size=16),
        factors={"A": np.tile([-1, 1], 8), "B": np.repeat([-1, 1], 8)},
    )


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 50)
    return EDAData(rng.normal(loc=x), x=x)


# --- Construction ---

class TestEDADataConstruction:
    def test_y_only(self):
        data = EDAData([1.0, 2.0, 3.0])
        assert isinstance(data.y, np.ndarray)
        assert data.x is None
        assert data.t is None
        assert data.factors is None

    def test_with_x(self):
        data = EDAData([1.0, 2.0, 3.0], x=[4.0, 5.0, 6.0])
        assert isinstance(data.x, np.ndarray)
        np.testing.assert_array_equal(data.x, [4.0, 5.0, 6.0])

    def test_with_t(self):
        data = EDAData([1.0, 2.0, 3.0], t=[0.1, 0.2, 0.3])
        assert isinstance(data.t, np.ndarray)
        assert data.t.dtype == float

    def test_with_factors(self):
        data = EDAData([1.0, 2.0, 3.0], factors={"A": [1, 2, 3]})
        assert isinstance(data.factors, dict)
        assert isinstance(data.factors["A"], np.ndarray)

    def test_list_inputs_stored_as_ndarray(self):
        data = EDAData([1, 2, 3])
        assert isinstance(data.y, np.ndarray)

    def test_t_stored_as_float(self):
        data = EDAData([1, 2, 3], t=[1, 2, 3])
        assert data.t.dtype == float


# --- Validation ---

class TestEDADataValidation:
    def test_empty_y_raises(self):
        with pytest.raises(ValueError, match="y cannot be empty"):
            EDAData([])

    def test_multidim_y_raises(self):
        with pytest.raises(ValueError, match="y must be 1-dimensional"):
            EDAData(np.ones((3, 3)))

    def test_x_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="x and y must have the same length"):
            EDAData([1, 2, 3], x=[1, 2])

    def test_t_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="t and y must have the same length"):
            EDAData([1, 2, 3], t=[1, 2])

    def test_factors_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="factors\\['A'\\] and y must have"):
            EDAData([1, 2, 3], factors={"A": [1, 2]})

    def test_multidim_x_raises(self):
        with pytest.raises(ValueError, match="x must be 1-dimensional"):
            EDAData([1, 2, 3, 4], x=np.ones((2, 2)))

    def test_multidim_t_raises(self):
        with pytest.raises(ValueError, match="t must be 1-dimensional"):
            EDAData([1, 2, 3, 4], t=np.ones((2, 2)))

    def test_multidim_factor_raises(self):
        with pytest.raises(ValueError, match="factors\\['A'\\] must be 1-dimensional"):
            EDAData([1, 2, 3, 4], factors={"A": np.ones((2, 2))})
```

- [ ] **Step 1.2: Run tests to verify they fail**

```
uv run pytest tests/test_data.py -v
```

Expected: `ModuleNotFoundError: No module named 'drippy.data'`

- [ ] **Step 1.3: Implement `src/drippy/data.py`**

```python
"""EDA data container."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class EDAData:
    """Validated data container for EDA analysis.

    Args:
        y: Response variable. Must be 1D and non-empty.
        x: Continuous predictor or single categorical factor.
            Must match len(y) if provided.
        t: Continuous index variable (e.g. time, 1/B, position).
            Not restricted to real time. Must match len(y) if provided.
        factors: Named factor arrays for multi-factor/DOE/comparative
            plots. Each value must match len(y).
    """

    def __init__(
        self,
        y: Iterable[float],
        x: Iterable | None = None,
        t: Iterable[float] | None = None,
        factors: dict[str, Iterable] | None = None,
    ) -> None:
        self.y = np.asarray(y)
        if self.y.size == 0:
            msg = "y cannot be empty"
            raise ValueError(msg)
        if self.y.ndim != 1:
            msg = "y must be 1-dimensional"
            raise ValueError(msg)

        self.x = None
        if x is not None:
            self.x = np.asarray(x)
            if self.x.ndim != 1:
                msg = "x must be 1-dimensional"
                raise ValueError(msg)
            if len(self.x) != len(self.y):
                msg = (
                    f"x and y must have the same length. "
                    f"Got len(x)={len(self.x)}, len(y)={len(self.y)}."
                )
                raise ValueError(msg)

        self.t = None
        if t is not None:
            self.t = np.asarray(t, dtype=float)
            if self.t.ndim != 1:
                msg = "t must be 1-dimensional"
                raise ValueError(msg)
            if len(self.t) != len(self.y):
                msg = (
                    f"t and y must have the same length. "
                    f"Got len(t)={len(self.t)}, len(y)={len(self.y)}."
                )
                raise ValueError(msg)

        self.factors = None
        if factors is not None:
            self.factors = {}
            for key, val in factors.items():
                arr = np.asarray(val)
                if arr.ndim != 1:
                    msg = f"factors['{key}'] must be 1-dimensional"
                    raise ValueError(msg)
                if len(arr) != len(self.y):
                    msg = (
                        f"factors['{key}'] and y must have the same length. "
                        f"Got len(factors['{key}'])={len(arr)}, len(y)={len(self.y)}."
                    )
                    raise ValueError(msg)
                self.factors[key] = arr
```

- [ ] **Step 1.4: Run tests to verify they pass**

```
uv run pytest tests/test_data.py -v
```

Expected: all green.

- [ ] **Step 1.5: Commit**

```bash
git add src/drippy/data.py tests/test_data.py
git commit -m "feat: add EDAData validated data container"
```

---

## Task 2: Refactor univariate.py to functions

**Files:**
- Rewrite: `src/drippy/univariate.py`
- Rewrite: `tests/test_univariate.py`

- [ ] **Step 2.1: Write failing tests using the new functional API**

Fully replace `tests/test_univariate.py`:

```python
"""Tests for the drippy.univariate module."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mpl.use("Agg")

import drippy.univariate as uv  # noqa: E402
from drippy.data import EDAData  # noqa: E402


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
        np.testing.assert_array_equal(x_data, np.arange(len(univariate_data.y)))

    def test_uses_t_when_provided(self, univariate_data_with_t):
        _, ax = uv.run_sequence_plot(univariate_data_with_t)
        lines = ax.get_lines()
        x_data, _ = lines[0].get_data()
        np.testing.assert_array_equal(x_data, univariate_data_with_t.t)

    def test_custom_fig_ax(self, univariate_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = uv.run_sequence_plot(univariate_data, fig=provided_fig, ax=provided_ax)
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
        fig, ax = uv.lag_plot(univariate_data, fig=provided_fig, ax=provided_ax)
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
        np.testing.assert_array_equal(scatter_data[:, 0], univariate_data.y[:-lag])
        np.testing.assert_array_equal(scatter_data[:, 1], univariate_data.y[lag:])

    def test_negative_lag_raises(self, univariate_data):
        with pytest.raises(ValueError, match="Lag must be a positive integer"):
            uv.lag_plot(univariate_data, lag=-1)

    def test_zero_lag_raises(self, univariate_data):
        with pytest.raises(ValueError, match="Lag must be a positive integer"):
            uv.lag_plot(univariate_data, lag=0)

    def test_lag_too_large_raises(self, univariate_data):
        with pytest.raises(ValueError, match="Lag must be less than the length"):
            uv.lag_plot(univariate_data, lag=len(univariate_data.y))

    def test_float_lag_converted(self, univariate_data):
        fig, ax = uv.lag_plot(univariate_data, lag=2.7)
        assert isinstance(fig, Figure)


# --- histogram ---

class TestHistogram:
    def test_returns_figure_and_axes(self, univariate_data):
        fig, ax = uv.histogram(univariate_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, univariate_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = uv.histogram(univariate_data, fig=provided_fig, ax=provided_ax)
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
        fig, ax = uv.normal_probability_plot(univariate_data, fig=provided_fig, ax=provided_ax)
        assert fig is provided_fig
        assert ax is provided_ax

    def test_returns_rsquared_when_requested(self, univariate_data):
        result = uv.normal_probability_plot(univariate_data, return_rsquared=True)
        assert len(result) == 3
        fig, ax, rsq = result
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
        with pytest.raises(ValueError, match=r"Axes must be an iterable of \(2, 2\)"):
            uv.four_plot(univariate_data, fig=fig, axes=axes)

    def test_custom_fig_axes(self, univariate_data):
        provided_fig, provided_axes = plt.subplots(2, 2)
        fig, axes = uv.four_plot(univariate_data, fig=provided_fig, axes=provided_axes)
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
        with pytest.raises(ValueError, match="Rough range must contain exactly 2"):
            uv.ppcc_plot(univariate_data, rough_range=(0.1, 0.5, 0.9))

    def test_inverted_range_raises(self, univariate_data):
        with pytest.raises(ValueError, match="min < max"):
            uv.ppcc_plot(univariate_data, rough_range=(2, -2))

    def test_negative_n_raises(self, univariate_data):
        with pytest.raises(ValueError, match="Number of points must be positive"):
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
        with pytest.raises(ValueError, match=r"Axes must be an iterable of \(2, 2\)"):
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
```

- [ ] **Step 2.2: Run tests to verify they fail**

```
uv run pytest tests/test_univariate.py -v
```

Expected: `ImportError: cannot import name 'run_sequence_plot' from 'drippy.univariate'`

- [ ] **Step 2.3: Rewrite `src/drippy/univariate.py`**

```python
"""Plotting functions for univariate models (y = c + e)."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
from lmfit.models import LinearModel
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from drippy.data import EDAData


def run_sequence_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a run sequence plot of y vs index or continuous index t.

    Args:
        data: EDAData container. Uses t as x-axis if present, else index.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    fig, ax = get_figure_and_axes(fig, ax)
    x_vals = data.t if data.t is not None else np.arange(len(data.y))
    ax.plot(x_vals, data.y, label="Data")
    ax.legend()
    ax.set_xlabel("Index" if data.t is None else "t")
    ax.set_ylabel("Y")
    ax.set_title("Run Sequence Plot")
    fig.tight_layout()
    return fig, ax


def lag_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    lag: int = 1,
) -> tuple[Figure, Axes]:
    """Creates a lag plot of the data.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        lag: Number of lags. Must be positive and less than len(y).

    Returns:
        The figure and axes containing the plot.
    """
    lag = int(lag)
    if lag <= 0:
        msg = "Lag must be a positive integer"
        raise ValueError(msg)
    if lag >= len(data.y):
        msg = "Lag must be less than the length of the data"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    y_original = data.y[lag:]
    y_lagged = data.y[:-lag]
    colors = np.arange(len(y_lagged))
    scatter = ax.scatter(y_lagged, y_original, c=colors, cmap="viridis", label="Lag Plot")
    fig.colorbar(scatter, ax=ax, label="Index")
    ax.set_xlabel(rf"Y$_{{i-{lag}}}$")
    ax.set_ylabel("Y$_{i}$")
    ax.set_title(f"Lag Plot (lag={lag})")
    fig.tight_layout()
    return fig, ax


def histogram(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    bins: int | str = "auto",
) -> tuple[Figure, Axes]:
    """Creates a histogram of the data.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        bins: Number of bins or bin strategy.

    Returns:
        The figure and axes containing the plot.
    """
    if isinstance(bins, int) and bins <= 0:
        msg = f"Number of bins must be positive, got {bins}"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    ax.hist(data.y, bins=bins)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Y values")
    fig.tight_layout()
    return fig, ax


def normal_probability_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    return_rsquared: bool = False,
) -> tuple[Figure, Axes] | tuple[Figure, Axes, float]:
    """Creates a normal probability plot.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        return_rsquared: If True, returns R-squared as third element.

    Returns:
        (fig, ax) or (fig, ax, r_squared) if return_rsquared is True.
    """
    fig, ax = get_figure_and_axes(fig, ax)
    _, (_, _, r_value) = sp.stats.probplot(data.y, dist="norm", plot=ax, rvalue=True)
    ax.set_ylabel("Ordered Values")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_title("Normal Probability Plot")
    fig.tight_layout()
    if return_rsquared:
        return fig, ax, r_value**2
    return fig, ax


def four_plot(
    data: EDAData,
    fig: Figure | None = None,
    axes: Iterable[Axes] | None = None,
) -> tuple[Figure, Iterable[Axes]]:
    """Creates a 4-plot (run sequence, lag, histogram, normal probability).

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        axes: ndarray of Axes with shape (2, 2). If None, creates new axes.

    Returns:
        (fig, axes_flat) where axes_flat has shape (4,).
    """
    if fig is None or axes is None:
        fig, _ = get_figure_and_axes(fig, None)
        axes = fig.subplots(2, 2)
    if axes.shape != (2, 2):
        msg = "Axes must be an iterable of (2, 2) Axes objects."
        raise ValueError(msg)
    axes = axes.flatten()
    run_sequence_plot(data, fig, axes[0])
    lag_plot(data, fig, axes[1])
    histogram(data, fig, axes[2])
    normal_probability_plot(data, fig, axes[3])
    fig.tight_layout()
    return fig, axes


def ppcc_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Iterable[Axes] | None = None,
    rough_range: tuple[float, float] = (-2, 2),
    n_rough: int = 50,
    n_fine: int = 100,
) -> tuple[Figure, Iterable[Axes]]:
    """Creates a PPCC plot (rough + fine) for distribution shape estimation.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: ndarray of Axes with shape (2,). If None, creates new axes.
        rough_range: (min, max) range for rough search.
        n_rough: Points in rough plot.
        n_fine: Points in fine plot.

    Returns:
        (fig, axes) where axes has shape (2,).
    """
    if len(rough_range) != 2:  # noqa: PLR2004
        msg = "Rough range must contain exactly 2 elements"
        raise ValueError(msg)
    if rough_range[0] >= rough_range[1]:
        msg = "Rough range must be (min, max) with min < max"
        raise ValueError(msg)
    if n_rough <= 0:
        msg = "Number of points must be positive"
        raise ValueError(msg)
    if n_fine <= 0:
        msg = "Number of points must be positive"
        raise ValueError(msg)
    if fig is None or ax is None:
        fig, _ = get_figure_and_axes(fig, None)
        ax = fig.subplots(1, 2)
    if ax.shape != (2,):
        msg = "Axes must be an iterable of 2 Axes objects."
        raise ValueError(msg)
    rough_shape_values, rough_ppcc = sp.stats.ppcc_plot(
        data.y, rough_range[0], rough_range[1], N=n_rough, plot=ax[0],
    )
    rough_max_index = np.argmax(rough_ppcc)
    fine_shape_values, fine_ppcc = sp.stats.ppcc_plot(
        data.y,
        rough_shape_values[rough_max_index] - 0.5,
        rough_shape_values[rough_max_index] + 0.5,
        N=n_fine,
        plot=ax[1],
    )
    fine_max_index = np.argmax(fine_ppcc)
    max_shape = rough_shape_values[rough_max_index]
    ax[0].vlines(
        max_shape, 0, rough_ppcc[rough_max_index],
        color="r", label=f"Max PPCC at shape={max_shape:.3g}",
    )
    ax[0].legend()
    max_shape_fine = fine_shape_values[fine_max_index]
    ax[1].axvline(max_shape_fine, color="r", label=f"Max PPCC at shape={max_shape_fine:.3g}")
    ax[1].legend()
    ax[0].set_title("Rough PPCC Plot")
    ax[1].set_title("Fine PPCC Plot")
    fig.tight_layout()
    return fig, ax


def weibull_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a Weibull probability plot.

    Args:
        data: EDAData container. Requires y > 0.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if not np.all(data.y > 0):
        msg = "Weibull plot requires all y values to be positive."
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    n = len(data.y)
    p = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    weibull_probabilities = np.log(-np.log(1 - p))
    ordered_data = np.log(np.sort(data.y))
    model = LinearModel()
    params = model.make_params(intercept=0, slope=1)
    result = model.fit(weibull_probabilities, params, x=ordered_data)
    shape = result.params["slope"].value
    intercept = result.params["intercept"].value
    scale = np.exp(-intercept / shape)
    x_fit = np.linspace(min(ordered_data), max(ordered_data), 100)
    y_fit = intercept + shape * x_fit
    ax.plot(ordered_data, weibull_probabilities, ".", label="Data")
    ax.plot(x_fit, y_fit, "-", label=f"Fit with R$^2$={result.rsquared:.3g}")
    ax.set_xlabel("Ordered Values")
    ax.set_ylabel("Theoretical Quantiles (Weibull)")
    ax.set_title("Weibull Probability Plot")
    min_percentage = np.floor(
        np.log10(1 - np.exp(np.exp(min(weibull_probabilities)) * -1)),
    )
    percentages = np.concatenate(
        (np.logspace(min_percentage - 1, -1, num=int(-min_percentage) + 1), [0.5, 0.9, 0.99]),
    )
    ax.set_yticks(
        np.log(-np.log(1 - percentages)),
        labels=[f"{p * 100:.3g}" for p in percentages],
    )
    ax.axhline(np.log(-np.log(1 - 0.632)), color="black", linestyle="--")
    ax.axvline(np.log(scale), color="black", linestyle="--")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def probability_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    distribution: str = "norm",
) -> tuple[Figure, Axes]:
    """Creates a probability plot for a specified distribution.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        distribution: scipy.stats distribution name.

    Returns:
        The figure and axes containing the plot.
    """
    fig, ax = get_figure_and_axes(fig, ax)
    sp.stats.probplot(data.y, dist=distribution, plot=ax)
    ax.set_ylabel("Ordered Values")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_title(f"{distribution.capitalize()} Probability Plot")
    fig.tight_layout()
    return fig, ax


def box_cox_normality_plot(
    data: EDAData,
    fig: Figure | None = None,
    axes: Iterable[Axes] | None = None,
) -> tuple[Figure, Iterable[Axes]]:
    """Creates a Box-Cox normality plot (2x2 grid).

    Shows: original histogram, Box-Cox normality curve, transformed
    histogram, normal probability plot of transformed data.

    Args:
        data: EDAData container. Requires y > 0.
        fig: Matplotlib figure. If None, creates new figure.
        axes: ndarray of Axes with shape (2, 2). If None, creates new.

    Returns:
        (fig, axes) where axes has shape (2, 2).
    """
    if not np.all(data.y > 0):
        msg = "Box-Cox transformation requires all y values to be positive."
        raise ValueError(msg)
    if fig is None or axes is None:
        fig, axes = get_figure_and_axes(fig, None)
        axes = fig.subplots(2, 2)
    if axes.shape != (2, 2):
        msg = "Axes must be an iterable of (2, 2) Axes objects."
        raise ValueError(msg)
    sp.stats.boxcox_normplot(data.y, -2, 2, plot=axes[0, 1], N=200)
    transformed_y, maxlog = sp.stats.boxcox(data.y)
    axes[0, 0].hist(data.y, bins="auto")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Histogram of Original (positive shifted) Y values")
    axes[0, 1].set_title("Box-Cox Normality Plot")
    axes[0, 1].axvline(maxlog, color="r", linestyle="--", label=f"Max log={maxlog:.3g}")
    axes[0, 1].legend()
    axes[1, 0].hist(transformed_y, bins="auto")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Histogram of transformed Y values")
    sp.stats.probplot(transformed_y, dist="norm", plot=axes[1, 1], rvalue=True)
    axes[1, 1].set_title("Normal Probability Plot of Transformed Data")
    fig.tight_layout()
    return fig, axes


def bootstrap_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
) -> tuple[Figure, Axes]:
    """Creates a bootstrap distribution plot.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        statistic: Callable to bootstrap. Must be callable.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        The figure and axes containing the plot.
    """
    if not callable(statistic):
        msg = "Statistic must be callable"
        raise TypeError(msg)
    if n_bootstrap <= 0:
        msg = "Number of bootstrap samples must be positive"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    result = sp.stats.bootstrap((data.y,), statistic, n_resamples=n_bootstrap)
    variability_label = f"Variability of {statistic.__name__}: {result.standard_error:.3g}"
    ax.hist(result.bootstrap_distribution, bins="auto", density=True, label=variability_label)
    ax.set_xlabel(f"Bootstrap {statistic.__name__} values")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap Distribution")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def box_cox_linearity_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    lmbda_range: tuple[float, float] = (-2, 2),
    n: int = 100,
) -> tuple[Figure, Axes]:
    """Creates a Box-Cox linearity plot.

    Plots |corr(Y, X^λ)| across a range of λ values to find the
    power transformation of X that maximises linearity with Y.

    Args:
        data: EDAData container. Requires x > 0.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        lmbda_range: (min, max) range of λ to evaluate.
        n: Number of λ values to evaluate.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "box_cox_linearity_plot requires x"
        raise ValueError(msg)
    if not np.all(data.x > 0):
        msg = "box_cox_linearity_plot requires all x values to be positive"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    lambdas = np.linspace(lmbda_range[0], lmbda_range[1], n)
    correlations = np.array([
        abs(np.corrcoef(
            np.log(data.x) if lmbda == 0 else (data.x**lmbda - 1) / lmbda,
            data.y,
        )[0, 1])
        for lmbda in lambdas
    ])
    optimal_lambda = lambdas[np.argmax(correlations)]
    ax.plot(lambdas, correlations)
    ax.axvline(
        optimal_lambda, color="r", linestyle="--",
        label=f"Optimal λ={optimal_lambda:.3g}",
    )
    ax.set_xlabel("λ (power transformation)")
    ax.set_ylabel("|Correlation(Y, X^λ)|")
    ax.set_title("Box-Cox Linearity Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax
```

- [ ] **Step 2.4: Run tests to verify they pass**

```
uv run pytest tests/test_univariate.py -v
```

Expected: all green.

- [ ] **Step 2.5: Commit**

```bash
git add src/drippy/univariate.py tests/test_univariate.py
git commit -m "refactor: convert UnivariatePlotter to standalone functions"
```

---

## Task 3: Refactor timeseries.py to functions

**Files:**
- Rewrite: `src/drippy/timeseries.py`
- Rewrite: `tests/test_timeseries.py`

- [ ] **Step 3.1: Write failing tests using the new functional API**

Fully replace `tests/test_timeseries.py`:

```python
"""Tests for the drippy.timeseries module."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mpl.use("Agg")

import drippy.timeseries as ts  # noqa: E402
from drippy.data import EDAData  # noqa: E402
from drippy.univariate import run_sequence_plot  # noqa: E402


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
    def test_imported_from_univariate(self):
        import drippy.univariate as uv
        assert ts.run_sequence_plot is uv.run_sequence_plot

    def test_uses_t_when_provided(self, timeseries_data):
        _, ax = ts.run_sequence_plot(timeseries_data)
        lines = ax.get_lines()
        x_data, _ = lines[0].get_data()
        np.testing.assert_array_equal(x_data, timeseries_data.t)


# --- spectral_plot ---

class TestSpectralPlot:
    def test_returns_figure_and_axes(self, timeseries_data):
        fig, ax = ts.spectral_plot(timeseries_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_t(self, univariate_data):
        with pytest.raises(ValueError, match="requires t"):
            ts.spectral_plot(univariate_data)

    def test_custom_fig_ax(self, timeseries_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = ts.spectral_plot(timeseries_data, fig=provided_fig, ax=provided_ax)
        assert fig is provided_fig
        assert ax is provided_ax

    def test_alarm_levels_enabled(self, timeseries_data):
        _, ax = ts.spectral_plot(timeseries_data, alarm_levels=True)
        assert len(ax.get_lines()) >= 1

    def test_alarm_levels_disabled(self, timeseries_data):
        _, ax = ts.spectral_plot(timeseries_data, alarm_levels=False)
        assert len(ax.get_lines()) >= 1

    def test_has_labels(self, timeseries_data):
        _, ax = ts.spectral_plot(timeseries_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""


# --- autocorrelation_plot ---

class TestAutocorrelationPlot:
    def test_returns_figure_and_axes(self, timeseries_data):
        fig, ax = ts.autocorrelation_plot(timeseries_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_custom_fig_ax(self, timeseries_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = ts.autocorrelation_plot(timeseries_data, fig=provided_fig, ax=provided_ax)
        assert fig is provided_fig
        assert ax is provided_ax

    def test_creates_lines(self, timeseries_data):
        _, ax = ts.autocorrelation_plot(timeseries_data)
        assert len(ax.get_lines()) > 0

    def test_has_labels(self, timeseries_data):
        _, ax = ts.autocorrelation_plot(timeseries_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""


# --- complex_demodulation_amplitude_plot ---

class TestComplexDemodulationAmplitudePlot:
    def test_returns_figure_and_axes(self, timeseries_data):
        fig, ax = ts.complex_demodulation_amplitude_plot(timeseries_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_t(self, univariate_data):
        with pytest.raises(ValueError, match="requires t"):
            ts.complex_demodulation_amplitude_plot(univariate_data)

    def test_custom_fig_ax(self, timeseries_data):
        provided_fig, provided_ax = plt.subplots()
        fig, ax = ts.complex_demodulation_amplitude_plot(
            timeseries_data, fig=provided_fig, ax=provided_ax
        )
        assert fig is provided_fig
        assert ax is provided_ax

    def test_has_labels(self, timeseries_data):
        _, ax = ts.complex_demodulation_amplitude_plot(timeseries_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""


# --- complex_demodulation_phase_plot ---

class TestComplexDemodulationPhasePlot:
    def test_returns_figure_and_axes(self, timeseries_data):
        fig, ax = ts.complex_demodulation_phase_plot(timeseries_data)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_requires_t(self, univariate_data):
        with pytest.raises(ValueError, match="requires t"):
            ts.complex_demodulation_phase_plot(univariate_data)

    def test_creates_two_lines(self, timeseries_data):
        _, ax = ts.complex_demodulation_phase_plot(timeseries_data)
        assert len(ax.get_lines()) >= 2

    def test_has_labels(self, timeseries_data):
        _, ax = ts.complex_demodulation_phase_plot(timeseries_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""
```

- [ ] **Step 3.2: Run tests to verify they fail**

```
uv run pytest tests/test_timeseries.py -v
```

Expected: `ImportError: cannot import name 'spectral_plot' from 'drippy.timeseries'`

- [ ] **Step 3.3: Rewrite `src/drippy/timeseries.py`**

```python
"""Plotting functions for time series models (y = f(t) + e)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
from astropy.timeseries import LombScargle
from lmfit.models import LinearModel
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from drippy.univariate import run_sequence_plot  # shared function
from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from drippy.data import EDAData

__all__ = [
    "run_sequence_plot",
    "spectral_plot",
    "autocorrelation_plot",
    "complex_demodulation_amplitude_plot",
    "complex_demodulation_phase_plot",
]


def spectral_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    alarm_levels: bool = True,
) -> tuple[Figure, Axes]:
    """Creates a Lomb-Scargle periodogram.

    Args:
        data: EDAData container. Requires t.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
        alarm_levels: Whether to show false alarm probability levels.

    Returns:
        The figure and axes containing the plot.
    """
    if data.t is None:
        msg = "spectral_plot requires t (continuous index)"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    ls = LombScargle(data.t, data.y)
    frequency, power = ls.autopower(normalization="psd")
    ax.plot(frequency, power, label="Data")
    if alarm_levels:
        false_alarm_max_peak = ls.false_alarm_probability(power.max())
        false_alarm_levels = ls.false_alarm_level([0.1, 0.05, 0.01])
        ax.plot(
            frequency[np.argmax(power)],
            power.max(),
            "rx",
            label=f"False alarm level max peak: {false_alarm_max_peak * 100:.3g}%",
        )
        for i, fal in enumerate(false_alarm_levels):
            ax.axhline(
                fal,
                color=f"C{i + 1}",
                linestyle="--",
                label=f"False Alarm Level {['10%', '5%', '1%'][i]}",
            )
    ax.legend()
    ax.set_xlabel("Frequency (cycles per unit t)")
    ax.set_ylabel("Spectral Power Density")
    ax.set_title("Lomb-Scargle Periodogram")
    fig.tight_layout()
    return fig, ax


def autocorrelation_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates autocorrelation plot with 99%, 95%, and 80% confidence intervals.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    fig, ax = get_figure_and_axes(fig, ax)
    n = len(data.y)
    ax.acorr(data.y, usevlines=True, maxlags=n - 1)
    for i, ci in enumerate([0.99, 0.95, 0.8]):
        conf_level = sp.stats.norm.ppf((1 + ci) / 2) / np.sqrt(n)
        ax.axhline(
            conf_level, color=f"C{i + 1}", linestyle="--",
            label=f"{ci * 100:.0f}% confidence level",
        )
        ax.axhline(-conf_level, color=f"C{i + 1}", linestyle="--")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def complex_demodulation_amplitude_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates instantaneous amplitude plot via Hilbert transform.

    Args:
        data: EDAData container. Requires t.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.t is None:
        msg = "complex_demodulation_amplitude_plot requires t (continuous index)"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    analytic_signal = sp.signal.hilbert(data.y)
    instantaneous_amplitude = np.abs(analytic_signal)
    ax.plot(data.t, instantaneous_amplitude, label="Instantaneous Amplitude")
    ax.set_xlabel("t")
    ax.set_ylabel("Amplitude")
    ax.set_title("Complex Demodulation Amplitude Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def complex_demodulation_phase_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates instantaneous phase plot via Hilbert transform with linear fit.

    Args:
        data: EDAData container. Requires t.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.t is None:
        msg = "complex_demodulation_phase_plot requires t (continuous index)"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    analytic_signal = sp.signal.hilbert(data.y)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    model = LinearModel()
    params = model.make_params(intercept=0, slope=0)
    result = model.fit(instantaneous_phase, params, x=data.t)
    ax.plot(data.t, instantaneous_phase, label="Instantaneous Phase")
    ax.plot(
        data.t,
        result.best_fit,
        color="red",
        label=(
            f"Linear Fit with R$^2$={result.rsquared:.3g}\n"
            rf"$\phi_0$={result.params['intercept'].value:.3g}, "
            rf"$\omega$={result.params['slope'].value:.3g}"
        ),
    )
    ax.set_xlabel("t")
    ax.set_ylabel("Phase (radians)")
    ax.set_title("Complex Demodulation Phase Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax
```

- [ ] **Step 3.4: Run tests to verify they pass**

```
uv run pytest tests/test_timeseries.py -v
```

Expected: all green.

- [ ] **Step 3.5: Commit**

```bash
git add src/drippy/timeseries.py tests/test_timeseries.py
git commit -m "refactor: convert TimeSeriesPlotter to standalone functions"
```

---

## Task 4: Create onefactor.py

**Files:**
- Create: `src/drippy/onefactor.py`
- Create: `tests/test_onefactor.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_onefactor.py`:

```python
"""Tests for the drippy.onefactor module."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mpl.use("Agg")

import drippy.onefactor as of  # noqa: E402
from drippy.data import EDAData  # noqa: E402


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
        fig, ax = of.scatter_plot(onefactor_data, fig=provided_fig, ax=provided_ax)
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
        # matplotlib boxplot produces 6 Line2D artists per box
        # (2 whiskers, 2 caps, 1 median, 1 box outline drawn as lines)
        assert len(ax.get_lines()) == n_levels * 6

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
```

- [ ] **Step 4.2: Run tests to verify they fail**

```
uv run pytest tests/test_onefactor.py -v
```

Expected: `ModuleNotFoundError: No module named 'drippy.onefactor'`

- [ ] **Step 4.3: Implement `src/drippy/onefactor.py`**

```python
"""Plotting functions for 1-factor models (y = f(x) + e, x categorical)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from drippy.utilities import get_figure_and_axes

if TYPE_CHECKING:
    from drippy.data import EDAData


def scatter_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a scatter plot of y vs x.

    Also used in regression context (see drippy.regression).

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "scatter_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    ax.scatter(data.x, data.y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter Plot")
    fig.tight_layout()
    return fig, ax


def box_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a box plot of y grouped by factor levels in x.

    Args:
        data: EDAData container. Requires x (categorical).
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "box_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    levels = np.unique(data.x)
    groups = [data.y[data.x == level] for level in levels]
    ax.boxplot(groups, tick_labels=levels)
    ax.set_xlabel("Factor Level")
    ax.set_ylabel("Y")
    ax.set_title("Box Plot")
    fig.tight_layout()
    return fig, ax


def bihistogram(
    data: EDAData,
    fig: Figure | None = None,
    axes: object | None = None,
    bins: int | str = "auto",
) -> tuple[Figure, object]:
    """Creates side-by-side histograms for exactly 2 factor levels.

    Args:
        data: EDAData container. Requires x with exactly 2 unique levels.
        fig: Matplotlib figure. If None, creates new figure.
        axes: Array of 2 Axes. If None, creates new axes.
        bins: Number of bins or bin strategy.

    Returns:
        (fig, axes) where axes is a 1-D array of 2 Axes.
    """
    if data.x is None:
        msg = "bihistogram requires x"
        raise ValueError(msg)
    levels = np.unique(data.x)
    if len(levels) != 2:  # noqa: PLR2004
        msg = f"bihistogram requires exactly 2 factor levels, got {len(levels)}"
        raise ValueError(msg)
    if fig is None or axes is None:
        fig, _ = get_figure_and_axes(fig, None)
        axes = fig.subplots(1, 2)
    group_a = data.y[data.x == levels[0]]
    group_b = data.y[data.x == levels[1]]
    axes[0].hist(group_a, bins=bins)
    axes[0].set_title(f"Histogram: {levels[0]}")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(group_b, bins=bins)
    axes[1].set_title(f"Histogram: {levels[1]}")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    fig.tight_layout()
    return fig, axes


def qq_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a quantile-quantile plot comparing 2 factor level distributions.

    Args:
        data: EDAData container. Requires x with exactly 2 unique levels.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "qq_plot requires x"
        raise ValueError(msg)
    levels = np.unique(data.x)
    if len(levels) != 2:  # noqa: PLR2004
        msg = f"qq_plot requires exactly 2 factor levels, got {len(levels)}"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    group_a = data.y[data.x == levels[0]]
    group_b = data.y[data.x == levels[1]]
    n = min(len(group_a), len(group_b))
    quantiles = np.linspace(0, 1, n)
    qa = np.quantile(group_a, quantiles)
    qb = np.quantile(group_b, quantiles)
    ax.scatter(qa, qb, label="Quantiles")
    min_val = min(qa.min(), qb.min())
    max_val = max(qa.max(), qb.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
    ax.set_xlabel(f"Quantiles: {levels[0]}")
    ax.set_ylabel(f"Quantiles: {levels[1]}")
    ax.set_title("Quantile-Quantile Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def mean_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a mean plot of y grouped by factor levels in x.

    Shows group means connected by a line, with a horizontal grand mean.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "mean_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    levels = np.unique(data.x)
    means = [data.y[data.x == level].mean() for level in levels]
    ax.plot(levels, means, "o-")
    ax.axhline(data.y.mean(), color="r", linestyle="--", label="Grand mean")
    ax.set_xlabel("Factor Level")
    ax.set_ylabel("Mean of Y")
    ax.set_title("Mean Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def sd_plot(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a standard deviation plot of y grouped by factor levels in x.

    Shows group standard deviations connected by a line, with a horizontal
    overall standard deviation reference line.

    Args:
        data: EDAData container. Requires x.
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.

    Returns:
        The figure and axes containing the plot.
    """
    if data.x is None:
        msg = "sd_plot requires x"
        raise ValueError(msg)
    fig, ax = get_figure_and_axes(fig, ax)
    levels = np.unique(data.x)
    sds = [data.y[data.x == level].std() for level in levels]
    ax.plot(levels, sds, "o-")
    ax.axhline(data.y.std(), color="r", linestyle="--", label="Overall SD")
    ax.set_xlabel("Factor Level")
    ax.set_ylabel("Standard Deviation of Y")
    ax.set_title("Standard Deviation Plot")
    ax.legend()
    fig.tight_layout()
    return fig, ax
```

- [ ] **Step 4.4: Run tests to verify they pass**

```
uv run pytest tests/test_onefactor.py -v
```

Expected: all green.

- [ ] **Step 4.5: Commit**

```bash
git add src/drippy/onefactor.py tests/test_onefactor.py
git commit -m "feat: add onefactor plotting module (scatter, box, bihistogram, qq, mean, sd)"
```

---

## Task 5: Add fluent methods to EDAData

**Files:**
- Modify: `src/drippy/data.py`
- Modify: `tests/test_data.py` (append fluent method tests)

- [ ] **Step 5.1: Write failing fluent method tests**

Append to `tests/test_data.py`:

```python
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
        fig, ax = univariate_data.normal_probability_plot()
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
        fig, ax = univariate_data.histogram(bins=5)
        assert len(ax.patches) == 5

    def test_future_module_raises_import_error(self, univariate_data):
        """Phase 2+ methods raise ImportError until those modules exist."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            univariate_data.doe_scatter_plot()
```

- [ ] **Step 5.2: Run tests to verify they fail**

```
uv run pytest tests/test_data.py::TestFluentMethods -v
```

Expected: `AttributeError: 'EDAData' object has no attribute 'run_sequence_plot'`

- [ ] **Step 5.3: Add fluent methods to `src/drippy/data.py`**

Append the following methods to the `EDAData` class (inside the class body, after `__init__`):

```python
    # --- Fluent methods (univariate) ---

    def run_sequence_plot(self, **kwargs):
        """Delegates to drippy.univariate.run_sequence_plot."""
        from drippy.univariate import run_sequence_plot
        return run_sequence_plot(self, **kwargs)

    def lag_plot(self, **kwargs):
        """Delegates to drippy.univariate.lag_plot."""
        from drippy.univariate import lag_plot
        return lag_plot(self, **kwargs)

    def histogram(self, **kwargs):
        """Delegates to drippy.univariate.histogram."""
        from drippy.univariate import histogram
        return histogram(self, **kwargs)

    def normal_probability_plot(self, **kwargs):
        """Delegates to drippy.univariate.normal_probability_plot."""
        from drippy.univariate import normal_probability_plot
        return normal_probability_plot(self, **kwargs)

    def four_plot(self, **kwargs):
        """Delegates to drippy.univariate.four_plot."""
        from drippy.univariate import four_plot
        return four_plot(self, **kwargs)

    def ppcc_plot(self, **kwargs):
        """Delegates to drippy.univariate.ppcc_plot."""
        from drippy.univariate import ppcc_plot
        return ppcc_plot(self, **kwargs)

    def weibull_plot(self, **kwargs):
        """Delegates to drippy.univariate.weibull_plot."""
        from drippy.univariate import weibull_plot
        return weibull_plot(self, **kwargs)

    def probability_plot(self, **kwargs):
        """Delegates to drippy.univariate.probability_plot."""
        from drippy.univariate import probability_plot
        return probability_plot(self, **kwargs)

    def box_cox_linearity_plot(self, **kwargs):
        """Delegates to drippy.univariate.box_cox_linearity_plot."""
        from drippy.univariate import box_cox_linearity_plot
        return box_cox_linearity_plot(self, **kwargs)

    def box_cox_normality_plot(self, **kwargs):
        """Delegates to drippy.univariate.box_cox_normality_plot."""
        from drippy.univariate import box_cox_normality_plot
        return box_cox_normality_plot(self, **kwargs)

    def bootstrap_plot(self, **kwargs):
        """Delegates to drippy.univariate.bootstrap_plot."""
        from drippy.univariate import bootstrap_plot
        return bootstrap_plot(self, **kwargs)

    # --- Fluent methods (timeseries) ---

    def spectral_plot(self, **kwargs):
        """Delegates to drippy.timeseries.spectral_plot."""
        from drippy.timeseries import spectral_plot
        return spectral_plot(self, **kwargs)

    def autocorrelation_plot(self, **kwargs):
        """Delegates to drippy.timeseries.autocorrelation_plot."""
        from drippy.timeseries import autocorrelation_plot
        return autocorrelation_plot(self, **kwargs)

    def complex_demodulation_amplitude_plot(self, **kwargs):
        """Delegates to drippy.timeseries.complex_demodulation_amplitude_plot."""
        from drippy.timeseries import complex_demodulation_amplitude_plot
        return complex_demodulation_amplitude_plot(self, **kwargs)

    def complex_demodulation_phase_plot(self, **kwargs):
        """Delegates to drippy.timeseries.complex_demodulation_phase_plot."""
        from drippy.timeseries import complex_demodulation_phase_plot
        return complex_demodulation_phase_plot(self, **kwargs)

    # --- Fluent methods (onefactor) ---

    def scatter_plot(self, **kwargs):
        """Delegates to drippy.onefactor.scatter_plot."""
        from drippy.onefactor import scatter_plot
        return scatter_plot(self, **kwargs)

    def box_plot(self, **kwargs):
        """Delegates to drippy.onefactor.box_plot."""
        from drippy.onefactor import box_plot
        return box_plot(self, **kwargs)

    def bihistogram(self, **kwargs):
        """Delegates to drippy.onefactor.bihistogram."""
        from drippy.onefactor import bihistogram
        return bihistogram(self, **kwargs)

    def qq_plot(self, **kwargs):
        """Delegates to drippy.onefactor.qq_plot."""
        from drippy.onefactor import qq_plot
        return qq_plot(self, **kwargs)

    def mean_plot(self, **kwargs):
        """Delegates to drippy.onefactor.mean_plot."""
        from drippy.onefactor import mean_plot
        return mean_plot(self, **kwargs)

    def sd_plot(self, **kwargs):
        """Delegates to drippy.onefactor.sd_plot."""
        from drippy.onefactor import sd_plot
        return sd_plot(self, **kwargs)

    # --- Fluent methods (Phase 2 — multifactor) ---

    def doe_scatter_plot(self, **kwargs):
        """Delegates to drippy.multifactor.doe_scatter_plot (Phase 2)."""
        from drippy.multifactor import doe_scatter_plot
        return doe_scatter_plot(self, **kwargs)

    def doe_mean_plot(self, **kwargs):
        """Delegates to drippy.multifactor.doe_mean_plot (Phase 2)."""
        from drippy.multifactor import doe_mean_plot
        return doe_mean_plot(self, **kwargs)

    def doe_sd_plot(self, **kwargs):
        """Delegates to drippy.multifactor.doe_sd_plot (Phase 2)."""
        from drippy.multifactor import doe_sd_plot
        return doe_sd_plot(self, **kwargs)

    def contour_plot(self, **kwargs):
        """Delegates to drippy.multifactor.contour_plot (Phase 2)."""
        from drippy.multifactor import contour_plot
        return contour_plot(self, **kwargs)

    # --- Fluent methods (Phase 3 — regression) ---

    def six_plot(self, **kwargs):
        """Delegates to drippy.regression.six_plot (Phase 3)."""
        from drippy.regression import six_plot
        return six_plot(self, **kwargs)

    def linear_correlation_plot(self, **kwargs):
        """Delegates to drippy.regression.linear_correlation_plot (Phase 3)."""
        from drippy.regression import linear_correlation_plot
        return linear_correlation_plot(self, **kwargs)

    def linear_intercept_plot(self, **kwargs):
        """Delegates to drippy.regression.linear_intercept_plot (Phase 3)."""
        from drippy.regression import linear_intercept_plot
        return linear_intercept_plot(self, **kwargs)

    def linear_slope_plot(self, **kwargs):
        """Delegates to drippy.regression.linear_slope_plot (Phase 3)."""
        from drippy.regression import linear_slope_plot
        return linear_slope_plot(self, **kwargs)

    def linear_residual_sd_plot(self, **kwargs):
        """Delegates to drippy.regression.linear_residual_sd_plot (Phase 3)."""
        from drippy.regression import linear_residual_sd_plot
        return linear_residual_sd_plot(self, **kwargs)

    # --- Fluent methods (Phase 4 — comparative) ---

    def block_plot(self, **kwargs):
        """Delegates to drippy.comparative.block_plot (Phase 4)."""
        from drippy.comparative import block_plot
        return block_plot(self, **kwargs)

    def youden_plot(self, **kwargs):
        """Delegates to drippy.comparative.youden_plot (Phase 4)."""
        from drippy.comparative import youden_plot
        return youden_plot(self, **kwargs)

    def star_plot(self, **kwargs):
        """Delegates to drippy.comparative.star_plot (Phase 4)."""
        from drippy.comparative import star_plot
        return star_plot(self, **kwargs)
```

- [ ] **Step 5.4: Run tests to verify they pass**

```
uv run pytest tests/test_data.py -v
```

Expected: all green.

- [ ] **Step 5.5: Commit**

```bash
git add src/drippy/data.py tests/test_data.py
git commit -m "feat: add fluent plotting methods to EDAData via lazy imports"
```

---

## Task 6: Update __init__.py and verify full suite

**Files:**
- Rewrite: `src/drippy/__init__.py`

- [ ] **Step 6.1: Rewrite `src/drippy/__init__.py`**

```python
"""drippy — EDA plotting library following NIST/SEMATECH principles."""

import logging

from drippy.data import EDAData
from drippy.onefactor import (
    bihistogram,
    box_plot,
    mean_plot,
    qq_plot,
    scatter_plot,
    sd_plot,
)
from drippy.timeseries import (
    autocorrelation_plot,
    complex_demodulation_amplitude_plot,
    complex_demodulation_phase_plot,
    spectral_plot,
)
from drippy.univariate import (
    bootstrap_plot,
    box_cox_linearity_plot,
    box_cox_normality_plot,
    four_plot,
    histogram,
    lag_plot,
    normal_probability_plot,
    ppcc_plot,
    probability_plot,
    run_sequence_plot,
    weibull_plot,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Michiel Dubbelman"
__email__ = "m.p.dubbelman@tudelft.nl"
__version__ = "0.1.0"

__all__ = [
    "EDAData",
    # univariate
    "run_sequence_plot",
    "lag_plot",
    "histogram",
    "normal_probability_plot",
    "four_plot",
    "ppcc_plot",
    "weibull_plot",
    "probability_plot",
    "box_cox_linearity_plot",
    "box_cox_normality_plot",
    "bootstrap_plot",
    # timeseries
    "spectral_plot",
    "autocorrelation_plot",
    "complex_demodulation_amplitude_plot",
    "complex_demodulation_phase_plot",
    # onefactor
    "scatter_plot",
    "box_plot",
    "bihistogram",
    "qq_plot",
    "mean_plot",
    "sd_plot",
]
```

- [ ] **Step 6.2: Run the full test suite**

```
uv run pytest -v
```

Expected: all tests pass. If any fail, fix before committing.

- [ ] **Step 6.3: Run linter**

```
uv run ruff check .
uv run ruff format --check .
```

Fix any issues, then rerun until clean.

- [ ] **Step 6.4: Commit**

```bash
git add src/drippy/__init__.py
git commit -m "feat: update __init__ exports for functional API (Phase 1 complete)"
```

---

## Phase 2 Pickup Note

When starting Phase 2 (`multifactor.py`), read:
- `docs/superpowers/specs/2026-04-14-nist-eda-plots-design.md` — full architecture
- `src/drippy/onefactor.py` — pattern to follow exactly
- `tests/test_onefactor.py` — fixture/test pattern to replicate

Phase 2 adds: `doe_scatter_plot`, `doe_mean_plot`, `doe_sd_plot`, `contour_plot` (all require `data.factors`). The `EDAData` fluent stubs already exist in `data.py` — no changes to `data.py` needed.
