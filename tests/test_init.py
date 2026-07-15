"""Tests for the top-level drippy package exports."""

from __future__ import annotations
import matplotlib as mpl

mpl.use("Agg")

import drippy


class TestPublicExports:
    """Every name in drippy.__all__ must be importable from the package."""

    def test_all_names_are_accessible(self):
        for name in drippy.__all__:
            assert hasattr(drippy, name), f"drippy.{name} is not accessible"

    def test_all_names_are_not_none(self):
        for name in drippy.__all__:
            assert getattr(drippy, name) is not None
