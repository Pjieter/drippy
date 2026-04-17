# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# New session and subagent policy
Always start the new session with running the /caveman full skill. Any subagent should also first run /caveman full.

## Code Exploration Policy

Always use jCodemunch-MCP tools — never fall back to Read, Grep, Glob, or Bash for code exploration.
- Before reading a file: use `get_file_outline` or `get_file_content`
- Before searching: use `search_symbols` or `search_text`
- Before exploring structure: use `get_file_tree` or `get_repo_outline`
- Call `resolve_repo` with the current directory first; if not indexed, call `index_folder`.

## Project Overview

DRIPPY is a Python library for Exploratory Data Analysis (EDA) following NIST/SEMATECH principles. It provides plotter classes for statistical visualization and analysis.

## Commands

**Poetry is mandatory** — use it for all commands, not pip/venv directly.

```bash
poetry install --all-groups        # Install all dependencies

poetry run pytest -v               # Run all tests
poetry run pytest tests/test_timeseries.py::TestClass::test_name -v  # Single test
poetry run pytest -n auto          # Parallel test run (pytest-xdist)
poetry run coverage run && poetry run coverage report  # With coverage

poetry run ruff check .            # Lint
poetry run ruff format             # Format

# Docs
cd docs && sphinx-build -b html . _build/html   # Build docs
cd docs && make doctest                          # Test docstrings

# Version bumping (never edit manually)
poetry run bump-my-version bump patch   # 0.1.0 → 0.1.1
poetry run bump-my-version bump minor   # 0.1.0 → 0.2.0
poetry run bump-my-version bump major   # 0.1.0 → 1.0.0
```

`bump-my-version` updates version in: `pyproject.toml`, `src/drippy/__init__.py`, `CITATION.cff`, `docs/conf.py`.

## Architecture

```
src/drippy/
    data.py         — EDAData validated data container (fluent API entry point)
    univariate.py   — Standalone functions for univariate plots (y = c + e)
    timeseries.py   — Standalone functions for time series plots
    onefactor.py    — Standalone functions for 1-factor plots (y = f(x) + e)
    utilities.py    — Shared helpers (get_figure_and_axes, etc.)
tests/
    test_timeseries.py
    test_univariate.py
    test_utilities.py
```

All plotting functions follow the same pattern:
1. Accept `EDAData` container as first arg (validated on construction)
2. Accept optional `fig`/`ax` (or `axes` for multi-axes plots), delegate to `get_figure_and_axes()` from `utilities.py`
3. Always call `fig.tight_layout()` before returning `tuple[Figure, Axes]` (or `tuple[Figure, np.ndarray]` for multi-axes)

`EDAData` is a fluent wrapper: `EDAData(y=data).four_plot()` delegates to the standalone function.

## Code Style

- Line length: 79 characters (Ruff enforced)
- Docstrings: Google style (Napoleon)
- Type hints: required on all public functions and `__init__`
- Imports: force-single-line, isort-sorted
- Ruff `select = ["ALL"]` with specific ignores in `pyproject.toml`

## Testing Patterns

- Non-interactive backend at module top: `mpl.use("Agg")`
- Auto-close figures with `autouse=True` fixture to prevent memory leaks
- Group related tests in classes
- Test edge cases: empty data, mismatched lengths, multi-dimensional arrays
