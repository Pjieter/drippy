# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-10-25

### Changed

- Allow astropy 8: the runtime requirement is now `astropy>=7.1.1,<9` (was `<8`), so drippy installs next to astropy 8.x. The test suite passes against astropy 8.0.2
- Upgraded Ruff to 0.17.x and allowed ipykernel 7 in the docs group
- Refreshed all locked dependencies (49 packages, including astropy 8.0.2, python-dateutil 3.0.0 and sqlalchemy 2.1.6)
- The SonarCloud workflow skips Dependabot pull requests, which get no repository secrets and so always failed the scan
- Merging a version bump into `main` now releases it. `release.yml` runs on every push to `main` and, when tag `v<version>` does not exist yet, publishes to PyPI, tags the commit and creates the GitHub release from the version's changelog section. Creating the release by hand is no longer needed

### Security

- Upgraded the locked pillow to 13.0.0, certifi to 2026.10.23, filelock to 4.2.4 and virtualenv to 21.14.13, closing the remaining open Dependabot alerts. drippy does not pin any of these: pillow comes in through matplotlib, so users should upgrade pillow in their own environments. The other three are dev-only (certifi through Sphinx, filelock and virtualenv through tox)
- Upgraded the locked dev dependencies urllib3 to 2.8.0 and virtualenv to 21.14.5 (since superseded by 21.14.13 above), closing six Dependabot alerts (CVE-2026-97688, CVE-2026-97689, CVE-2026-102925, CVE-2026-102930, CVE-2026-102937, CVE-2026-102938). The installed package is unaffected: urllib3 reaches the lockfile only through Sphinx and virtualenv only through tox

## [0.2.0] - 2026-09-15

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

[Unreleased]: https://github.com/Pjieter/drippy/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Pjieter/drippy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Pjieter/drippy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Pjieter/drippy/releases/tag/v0.1.0
