"""EDA data container."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
        self._validate_y()
        self.x = self._validate_and_convert_x(x)
        self.t = self._validate_and_convert_t(t)
        self.factors = self._validate_and_convert_factors(factors)

    # --- Fluent methods (univariate) ---

    def run_sequence_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.univariate.run_sequence_plot."""
        from drippy.univariate import run_sequence_plot as fn

        return fn(self, **kwargs)

    def lag_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.univariate.lag_plot."""
        from drippy.univariate import lag_plot as fn

        return fn(self, **kwargs)

    def histogram(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.univariate.histogram."""
        from drippy.univariate import histogram as fn

        return fn(self, **kwargs)

    def normal_probability_plot(
        self, **kwargs: Any
    ) -> tuple[Figure, Axes, float | None]:
        """Delegates to drippy.univariate.normal_probability_plot."""
        from drippy.univariate import normal_probability_plot as fn

        return fn(self, **kwargs)

    def four_plot(self, **kwargs: Any) -> tuple[Figure, np.ndarray]:
        """Delegates to drippy.univariate.four_plot."""
        from drippy.univariate import four_plot as fn

        return fn(self, **kwargs)

    def ppcc_plot(self, **kwargs: Any) -> tuple[Figure, np.ndarray]:
        """Delegates to drippy.univariate.ppcc_plot."""
        from drippy.univariate import ppcc_plot as fn

        return fn(self, **kwargs)

    def weibull_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.univariate.weibull_plot."""
        from drippy.univariate import weibull_plot as fn

        return fn(self, **kwargs)

    def probability_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.univariate.probability_plot."""
        from drippy.univariate import probability_plot as fn

        return fn(self, **kwargs)

    def box_cox_linearity_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.univariate.box_cox_linearity_plot."""
        from drippy.univariate import box_cox_linearity_plot as fn

        return fn(self, **kwargs)

    def box_cox_normality_plot(
        self, **kwargs: Any
    ) -> tuple[Figure, np.ndarray]:
        """Delegates to drippy.univariate.box_cox_normality_plot."""
        from drippy.univariate import box_cox_normality_plot as fn

        return fn(self, **kwargs)

    def bootstrap_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.univariate.bootstrap_plot."""
        from drippy.univariate import bootstrap_plot as fn

        return fn(self, **kwargs)

    # --- Fluent methods (timeseries) ---

    def spectral_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.timeseries.spectral_plot."""
        from drippy.timeseries import spectral_plot as fn

        return fn(self, **kwargs)

    def autocorrelation_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.timeseries.autocorrelation_plot."""
        from drippy.timeseries import autocorrelation_plot as fn

        return fn(self, **kwargs)

    def complex_demodulation_amplitude_plot(
        self, **kwargs: Any
    ) -> tuple[Figure, Axes]:
        """Delegate to timeseries.complex_demodulation_amplitude_plot."""
        from drippy.timeseries import complex_demodulation_amplitude_plot as fn

        return fn(self, **kwargs)

    def complex_demodulation_phase_plot(
        self, **kwargs: Any
    ) -> tuple[Figure, Axes]:
        """Delegate to drippy.timeseries.complex_demodulation_phase_plot."""
        from drippy.timeseries import complex_demodulation_phase_plot as fn

        return fn(self, **kwargs)

    # --- Fluent methods (onefactor) ---

    def scatter_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.onefactor.scatter_plot."""
        from drippy.onefactor import scatter_plot as fn

        return fn(self, **kwargs)

    def box_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.onefactor.box_plot."""
        from drippy.onefactor import box_plot as fn

        return fn(self, **kwargs)

    def bihistogram(self, **kwargs: Any) -> tuple[Figure, np.ndarray]:
        """Delegates to drippy.onefactor.bihistogram."""
        from drippy.onefactor import bihistogram as fn

        return fn(self, **kwargs)

    def qq_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.onefactor.qq_plot."""
        from drippy.onefactor import qq_plot as fn

        return fn(self, **kwargs)

    def mean_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.onefactor.mean_plot."""
        from drippy.onefactor import mean_plot as fn

        return fn(self, **kwargs)

    def sd_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.onefactor.sd_plot."""
        from drippy.onefactor import sd_plot as fn

        return fn(self, **kwargs)

    # --- Fluent methods (Phase 2 — multifactor) ---

    def doe_scatter_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.multifactor.doe_scatter_plot (Phase 2)."""
        from drippy.multifactor import doe_scatter_plot as fn

        return fn(self, **kwargs)

    def doe_mean_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.multifactor.doe_mean_plot (Phase 2)."""
        from drippy.multifactor import doe_mean_plot as fn

        return fn(self, **kwargs)

    def doe_sd_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.multifactor.doe_sd_plot (Phase 2)."""
        from drippy.multifactor import doe_sd_plot as fn

        return fn(self, **kwargs)

    def contour_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.multifactor.contour_plot (Phase 2)."""
        from drippy.multifactor import contour_plot as fn

        return fn(self, **kwargs)

    # --- Fluent methods (Phase 3 — regression) ---

    def six_plot(
        self, **kwargs: Any
    ) -> tuple[Figure, tuple[Axes, Axes, Axes, Axes, Axes, Axes]]:
        """Delegates to drippy.regression.six_plot (Phase 3)."""
        from drippy.regression import six_plot as fn

        return fn(self, **kwargs)

    def linear_correlation_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegate to drippy.regression.linear_correlation_plot (Phase 3)."""
        from drippy.regression import linear_correlation_plot as fn

        return fn(self, **kwargs)

    def linear_intercept_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegate to drippy.regression.linear_intercept_plot (Phase 3)."""
        from drippy.regression import linear_intercept_plot as fn

        return fn(self, **kwargs)

    def linear_slope_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.regression.linear_slope_plot (Phase 3)."""
        from drippy.regression import linear_slope_plot as fn

        return fn(self, **kwargs)

    def linear_residual_sd_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegate to drippy.regression.linear_residual_sd_plot (Phase 3)."""
        from drippy.regression import linear_residual_sd_plot as fn

        return fn(self, **kwargs)

    # --- Fluent methods (Phase 4 — comparative) ---

    def block_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.comparative.block_plot (Phase 4)."""
        from drippy.comparative import block_plot as fn

        return fn(self, **kwargs)

    def youden_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.comparative.youden_plot (Phase 4)."""
        from drippy.comparative import youden_plot as fn

        return fn(self, **kwargs)

    def star_plot(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Delegates to drippy.comparative.star_plot (Phase 4)."""
        from drippy.comparative import star_plot as fn

        return fn(self, **kwargs)

    def _validate_y(self) -> None:
        """Validate y array."""
        if self.y.size == 0:
            msg = "y cannot be empty"
            raise ValueError(msg)
        if self.y.ndim != 1:
            msg = "y must be 1-dimensional"
            raise ValueError(msg)

    def _validate_and_convert_x(self, x: Iterable | None) -> np.ndarray | None:
        """Validate and convert x array."""
        if x is None:
            return None
        x_arr = np.asarray(x)
        if x_arr.ndim != 1:
            msg = "x must be 1-dimensional"
            raise ValueError(msg)
        if len(x_arr) != len(self.y):
            msg = (
                f"x and y must have the same length. "
                f"Got len(x)={len(x_arr)}, len(y)={len(self.y)}."
            )
            raise ValueError(msg)
        return x_arr

    def _validate_and_convert_t(
        self, t: Iterable[float] | None
    ) -> np.ndarray | None:
        """Validate and convert t array."""
        if t is None:
            return None
        t_arr = np.asarray(t, dtype=float)
        if t_arr.ndim != 1:
            msg = "t must be 1-dimensional"
            raise ValueError(msg)
        if len(t_arr) != len(self.y):
            msg = (
                f"t and y must have the same length. "
                f"Got len(t)={len(t_arr)}, len(y)={len(self.y)}."
            )
            raise ValueError(msg)
        return t_arr

    def _validate_and_convert_factors(
        self, factors: dict[str, Iterable] | None
    ) -> dict[str, np.ndarray] | None:
        """Validate and convert factors dict."""
        if factors is None:
            return None
        result = {}
        for key, val in factors.items():
            arr = np.asarray(val)
            if arr.ndim != 1:
                msg = f"factors['{key}'] must be 1-dimensional"
                raise ValueError(msg)
            if len(arr) != len(self.y):
                msg = (
                    f"factors['{key}'] and y must have the same length. "
                    f"Got len(factors['{key}'])={len(arr)}, "
                    f"len(y)={len(self.y)}."
                )
                raise ValueError(msg)
            result[key] = arr
        return result
