# NIST EDA Plots — Full Implementation Design

**Date:** 2026-04-14
**Scope:** Implement all 33 graphical techniques from NIST/SEMATECH e-Handbook of Statistical Methods, Chapter 1 (EDA), Section 3.3.
**Reference:** https://www.itl.nist.gov/div898/handbook/eda/section3/eda33.htm

---

## Goal

Replace the existing class-based architecture (`UnivariatePlotter`, `TimeSeriesPlotter`) with a functional API backed by a single validated data container (`EDAData`). Implement all 33 NIST EDA graphical techniques as standalone functions organized by EDA model type.

---

## Architecture

### Data Container: `EDAData`

Defined in `src/drippy/data.py`. Lightweight validated data bag — no plotting logic.

```python
class EDAData:
    def __init__(
        self,
        y: Iterable[float],
        x: Iterable | None = None,
        t: Iterable[float] | None = None,
        factors: dict[str, Iterable] | None = None,
    ) -> None:
```

**Fields:**
- `y` — response variable; always required
- `x` — continuous predictor or single categorical factor
- `t` — time index (time series plots)
- `factors` — named factors for multi-factor/DOE/comparative plots (`{"treatment": [...], "block": [...]}`)

**Validation at construction (raises `ValueError`):**
- `y` must be 1D and non-empty
- `x`, `t`, each value in `factors` must match `len(y)`
- All inputs stored as `np.ndarray` internally

**Usage by model:**
```python
EDAData(y)                               # univariate
EDAData(y, t=t)                          # time series
EDAData(y, x=factor)                     # 1-factor
EDAData(y, x=predictor)                  # regression
EDAData(y, factors={"A": a, "B": b})     # multi-factor/DOE
EDAData(y, x=y1, factors={"y2": y2})     # interlab (youden)
EDAData(y, factors={"v2": v2, "v3": v3}) # multivariate (star)
```

---

### Module Structure

```
src/drippy/
    __init__.py       — re-exports EDAData + all 35 plot functions
    data.py           — EDAData container
    univariate.py     — 11 functions (univariate model)
    timeseries.py     — 5 functions (time series model)
    onefactor.py      — 6 functions (1-factor model)
    multifactor.py    — 4 functions (multi-factor/screening model)
    regression.py     — 6 functions (regression model)
    comparative.py    — 3 functions (comparative/interlab/multivariate)
    utilities.py      — unchanged: get_figure_and_axes, bl_filt
tests/
    test_data.py          — EDAData validation (new)
    test_univariate.py    — rewritten (functions, not class methods)
    test_timeseries.py    — rewritten
    test_onefactor.py     — new
    test_multifactor.py   — new
    test_regression.py    — new
    test_comparative.py   — new
```

---

### Plot Function Signature

All plot functions share the same shape:

```python
def histogram(
    data: EDAData,
    fig: Figure | None = None,
    ax: Axes | None = None,
    bins: int | str = "auto",   # plot-specific kwargs
) -> tuple[Figure, Axes]:
    """Creates a histogram of the data.

    Args:
        data: EDAData container. Requires y.
        fig: Matplotlib figure. If None, creates new.
        ax: Matplotlib axes. If None, creates new.
        bins: Number of bins or bin strategy.

    Returns:
        The figure and axes containing the plot.
    """
```

- Always call `get_figure_and_axes(fig, ax)` from `utilities.py`
- Always call `fig.tight_layout()` before returning
- Return `tuple[Figure, Axes]` (composite plots return `tuple[Figure, Iterable[Axes]]`)

---

## Plot-to-Module Mapping

35 functions covering 33 handbook sections. `run_sequence_plot` and `scatter_plot` are single functions imported into multiple modules — not duplicated.

### `univariate.py` — model: y = c + e (11 plots)

| Function | NIST ref |
|----------|----------|
| `run_sequence_plot` | 1.3.3.25 |
| `lag_plot` | 1.3.3.15 |
| `histogram` | 1.3.3.14 |
| `normal_probability_plot` | 1.3.3.21 |
| `four_plot` | 1.3.3.32 |
| `ppcc_plot` | 1.3.3.23 |
| `weibull_plot` | 1.3.3.30 |
| `probability_plot` | 1.3.3.22 |
| `box_cox_linearity_plot` | 1.3.3.5 |
| `box_cox_normality_plot` | 1.3.3.6 |
| `bootstrap_plot` | 1.3.3.4 |

### `timeseries.py` — model: y = f(t) + e (5 plots)

| Function | NIST ref |
|----------|----------|
| `run_sequence_plot` | 1.3.3.25 (shared) |
| `spectral_plot` | 1.3.3.27 |
| `autocorrelation_plot` | 1.3.3.1 |
| `complex_demodulation_amplitude_plot` | 1.3.3.8 |
| `complex_demodulation_phase_plot` | 1.3.3.9 |

### `onefactor.py` — model: y = f(x) + e, x categorical (6 plots)

| Function | NIST ref |
|----------|----------|
| `scatter_plot` | 1.3.3.26 (shared) |
| `box_plot` | 1.3.3.7 |
| `bihistogram` | 1.3.3.2 |
| `qq_plot` | 1.3.3.24 |
| `mean_plot` | 1.3.3.20 |
| `sd_plot` | 1.3.3.28 |

All require `data.x`. `bihistogram` expects exactly 2 unique levels in `x`.

### `multifactor.py` — model: y = f(x1,x2,...,xk) + e (4 plots)

| Function | NIST ref |
|----------|----------|
| `doe_scatter_plot` | 1.3.3.11 |
| `doe_mean_plot` | 1.3.3.12 |
| `doe_sd_plot` | 1.3.3.13 |
| `contour_plot` | 1.3.3.10 |

All require `data.factors`. `contour_plot` accepts a `doe: bool = False` parameter for the DOE variant (1.3.3.10.1).

### `regression.py` — model: y = f(x1,x2,...,xk) + e (6 plots)

| Function | NIST ref |
|----------|----------|
| `scatter_plot` | 1.3.3.26 (shared) |
| `six_plot` | 1.3.3.33 |
| `linear_correlation_plot` | 1.3.3.16 |
| `linear_intercept_plot` | 1.3.3.17 |
| `linear_slope_plot` | 1.3.3.18 |
| `linear_residual_sd_plot` | 1.3.3.19 |

All require `data.x`. Linear intercept/slope/residual-sd plots are typically used together on windowed/subset data — they accept a `window` parameter.

### `comparative.py` — comparative/interlab/multivariate (3 plots)

| Function | NIST model | NIST ref |
|----------|-----------|----------|
| `block_plot` | y = f(xp, x1…xk) + e | 1.3.3.3 |
| `youden_plot` | (y1,y2) = f(x) + e | 1.3.3.31 |
| `star_plot` | (y1,y2,…,yp) | 1.3.3.29 |

- `block_plot`: requires `data.factors` with keys `"treatment"` and `"block"`
- `youden_plot`: requires `data.y` (lab 1) and `data.x` (lab 2); accepts `doe: bool = False` for DOE variant (1.3.3.31.1)
- `star_plot`: requires `data.y` + `data.factors` for additional variables

---

## Error Handling

`EDAData.__init__` validates shape and length. Plot functions validate field presence only:

```python
def spectral_plot(data: EDAData, ...) -> tuple[Figure, Axes]:
    if data.t is None:
        raise ValueError("spectral_plot requires t (time index)")
```

No defensive checks beyond what `EDAData` already guarantees.

---

## Testing Patterns

```python
# Each test module follows this structure
import matplotlib as mpl
mpl.use("Agg")

import pytest
import numpy as np
import matplotlib.pyplot as plt
from drippy import EDAData
import drippy.onefactor as of

@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    y = rng.normal(size=50)
    x = np.repeat(["A", "B", "C", "D", "E"], 10)
    return EDAData(y, x=x)

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")

class TestBoxPlot:
    def test_returns_figure_and_axes(self, data): ...
    def test_custom_fig_ax(self, data): ...
    def test_requires_x(self):
        data = EDAData(np.ones(10))
        with pytest.raises(ValueError):
            of.box_plot(data)
```

`test_data.py` covers `EDAData` validation exhaustively: missing `y`, mismatched lengths, wrong dimensions, empty arrays.

---

## Migration from Class-Based API

Clean break — package is v0.1.0 pre-alpha, no deprecation shim needed.

| Old | New |
|-----|-----|
| `UnivariatePlotter(y, x).histogram()` | `drippy.histogram(EDAData(y))` |
| `TimeSeriesPlotter(y, t).spectral_plot()` | `drippy.spectral_plot(EDAData(y, t=t))` |

Existing test files (`test_univariate.py`, `test_timeseries.py`) are rewritten — class method calls become function calls.

---

## Implementation Order

1. **Phase 1 (this plan):** `data.py` + `onefactor.py` + `test_data.py` + `test_onefactor.py`. Refactor `univariate.py` and `timeseries.py` to functional style. Update `__init__.py`.
2. **Phase 2:** `multifactor.py` + `test_multifactor.py`
3. **Phase 3:** `regression.py` + `test_regression.py`
4. **Phase 4:** `comparative.py` + `test_comparative.py`
