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

**Poetry mandatory** — no pip/venv.

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

`bump-my-version` updates: `pyproject.toml`, `src/drippy/__init__.py`, `CITATION.cff`, `docs/conf.py`.

## Architecture

```
src/drippy/
    timeseries.py   — TimeSeriesPlotter class
    univariate.py   — UnivariatePlotter class
    utilities.py    — Shared helpers (get_figure_and_axes, etc.)
tests/
    test_timeseries.py
    test_univariate.py
    test_utilities.py
```

All plotters same pattern:
1. Validate inputs in `__init__` (empty arrays, dim mismatches, length equality)
2. Store as `np.asarray()` internally
3. Plotting methods accept optional `fig`/`ax`, delegate to `get_figure_and_axes()` from `utilities.py`
4. Always call `fig.tight_layout()` before return `tuple[Figure, Axes]`

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