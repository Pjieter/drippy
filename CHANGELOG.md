# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking:** minimum supported Python is now 3.12 (was 3.11); supported versions are 3.12, 3.13 and 3.14
- Multi-panel figures created by `doe_*` plots and `ppcc_plot` are now sized from the number of subplot columns, so `tight_layout` can fit every Axes instead of warning and giving up
- Upgraded Sphinx to 9.x, which requires Python 3.12+
- Upgraded Ruff to 0.16.x; `CPY001` (missing copyright notice) is ignored because licensing lives in `LICENSE` only
- Refreshed all locked dependencies (pytest 9.1.1, scipy 1.18.1, tox 4.61.4, urllib3 2.7.0, and others)
- Replaced Poetry with uv throughout `.github/copilot-instructions.md`, matching the toolchain the project actually uses
- Updated every GitHub Action to its current release: checkout 7.0.1, setup-uv 10.1.0, upload-artifact 7.0.1, download-artifact 8.0.1, lychee-action 2.9.0, create-issue-from-file 6.0.0 and sonarqube-scan-action 8.2.1. `setup-uv` is pinned to an exact version because it stopped publishing floating major tags at v8

### Added

- `drippy.utilities.get_grid_figsize()` for sizing a figure from its subplot grid shape

### Fixed

- `UserWarning: Tight layout not applied` raised by `doe_mean_plot()` and `ppcc_plot()` on default-sized figures

## [0.1.0] - 2026-08-29

### Added

- `EDAData` validated data container with a fluent plotting API
- 33 plotting functions implementing NIST/SEMATECH EDA techniques across univariate, time series, one-factor, multi-factor (DOE), regression, and comparative modules
- Sphinx documentation with example notebooks on ReadTheDocs
- PEP 561 `py.typed` marker (fully type-annotated public API)
- Published on PyPI as `drippy-eda` (the name `drippy` is taken by an unrelated package); import as `drippy`

[Unreleased]: https://github.com/Pjieter/drippy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Pjieter/drippy/releases/tag/v0.1.0
