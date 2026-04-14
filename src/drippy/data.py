"""EDA data container."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


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
