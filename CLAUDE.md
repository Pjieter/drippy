# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repo.

# New session and subagent policy

Always start new session: run /caveman full skill. Subagents: same.

## Code Exploration Policy

Use jCodemunch-MCP tools only — never Read, Grep, Glob, Bash for code exploration.
- Read file: use `get_file_outline` or `get_file_content`
- Search: use `search_symbols` or `search_text`
- Explore structure: use `get_file_tree` or `get_repo_outline`
- Call `resolve_repo` with current dir first; if not indexed, call `index_folder`.

## Project Overview

DRIPPY = Python EDA library, NIST/SEMATECH principles. Plotter classes for statistical viz/analysis.

## Commands

**Use `uv`/`uvx` — no pip/venv/python directly.**

```bash
uv sync --all-groups               # Install all dependencies

uv run pytest -v               # Run all tests
uv run pytest tests/test_timeseries.py::TestClass::test_name -v  # Single test
uv run pytest -n auto          # Parallel test run (pytest-xdist)
uv run coverage run && uv run coverage report  # With coverage

uv run ruff check .            # Lint
uv run ruff format             # Format

# Docs
cd docs && uv run sphinx-build -b html . _build/html   # Build docs
cd docs && uv run make doctest                          # Test docstrings

# Version bumping (never edit manually)
uv run bump-my-version bump patch   # 0.1.0 → 0.1.1
uv run bump-my-version bump minor   # 0.1.0 → 0.2.0
uv run bump-my-version bump major   # 0.1.0 → 1.0.0
```

`bump-my-version` updates: `pyproject.toml`, `src/drippy/__init__.py`, `CITATION.cff`, `docs/conf.py`.

## Architecture

```text
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

- Line length: 79 chars (Ruff enforced)
- Docstrings: Google style (Napoleon)
- Type hints: required all public funcs + `__init__`
- Imports: force-single-line, isort-sorted
- Ruff `select = ["ALL"]` — specific ignores in `pyproject.toml`

## Testing Patterns

- Non-interactive backend at module top: `mpl.use("Agg")`
- Auto-close figures via `autouse=True` fixture — prevent memory leaks
- Group related tests in classes
- Test edge cases: empty data, mismatched lengths, multi-dim arrays
