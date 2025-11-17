# DRIPPY - AI Coding Agent Instructions

## Project Overview

DRIPPY is a Python library for Exploratory Data Analysis (EDA) following NIST/SEMATECH principles ([NIST/SEMATECH e-Handbook](http://www.itl.nist.gov/div898/handbook/eda/eda.htm)). The package provides specialized plotter classes for statistical visualization and analysis.

**Core Architecture:**
- `src/drippy/` - Main package with plotter classes for different EDA models
- `src/drippy/utilities.py` - Shared utilities and helper functions used across plotters
- `tests/` - pytest-based test suite with fixtures and parametrization

## Development Setup

**Poetry is mandatory** - this project uses Poetry for dependency management, not pip/venv:

```powershell
# Install dependencies
poetry install --all-groups

# Run commands through Poetry
poetry run pytest -v
poetry run ruff check .
poetry run ruff format
```

**Python versions:** 3.11, 3.12, 3.13 (configured in `pyproject.toml`)

## Code Style Conventions

**Strict Ruff configuration** with line length 79 characters:
- Import style: Force single-line imports, sorted by `isort` integration
- Docstrings: Google style convention (Napoleon extension)
- Type hints: Required for all public functions (except dunder methods like `__init__`)
- No FBT (boolean trap), no explicit TODO markers in commits

**Example function signature:**

```python
def sequence_plot(
    self,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Creates a sequence plot showing data over time.
    
    Args:
        fig: Matplotlib figure. If None, creates new figure.
        ax: Matplotlib axes. If None, creates new axes.
    
    Returns:
        The figure and axes containing the plot.
    """
```

**Pattern for optional figure/axes parameters:**
All plotting methods follow the `get_figure_and_axes(fig, ax)` pattern from `utilities.py`, allowing users to provide existing figures or create new ones. Always call `fig.tight_layout()` before returning.

## Testing Patterns

**pytest with fixtures and parametrization:**
- Use `mpl.use("Agg")` at module top for non-interactive backend
- Create reusable `@pytest.fixture` for common data (see `test_timeseries.py`)
- Auto-close figures with `autouse=True` fixture to prevent memory leaks:

  ```python
  @pytest.fixture(autouse=True)
  def close_figures():
      yield
      plt.close("all")
  ```
- Group related tests in classes (e.g., `TestTimeSeriesPlotterInitialization`)
- Test edge cases: empty data, mismatched lengths, multi-dimensional arrays

**Run tests:**

```powershell
poetry run pytest -v           # Standard run
poetry run coverage run        # With coverage
poetry run coverage report     # View coverage
```

## Documentation

**Sphinx with AutoAPI and Google-style docstrings:**
- Source in `docs/` directory with RTD theme
- Build locally: `cd docs; sphinx-build -b html docs docs/_build/html`
- Test docstrings: `cd docs; make doctest`
- ReadTheDocs builds automatically from `.readthedocs.yaml`

## Plotter Class Patterns

When adding new plotter classes or plotting methods:
1. **Validate inputs in `__init__`**: Check for empty arrays, dimension mismatches, equal lengths
2. **Use numpy arrays internally**: Convert inputs with `np.asarray()` for consistency
3. **Provide meaningful error messages**: Include actual vs expected values in exceptions
4. **Optional figure/axes**: Always use `get_figure_and_axes()` helper from `utilities.py`
5. **Return tuple[Figure, Axes]**: Consistent return type across all plotting methods
6. **Call `fig.tight_layout()`**: Always call before returning to ensure proper spacing

## Version Bumping

Use `bump-my-version` (NOT manual editing):

```powershell
poetry run bump-my-version bump patch  # 0.1.0 -> 0.1.1
poetry run bump-my-version bump minor  # 0.1.0 -> 0.2.0
poetry run bump-my-version bump major  # 0.1.0 -> 1.0.0
```

Updates version in: `pyproject.toml`, `src/drippy/__init__.py`, `CITATION.cff`, `docs/conf.py`

**Dependencies**

**Core scientific stack:**
- `matplotlib` - Plotting (all methods return Figure/Axes)
- `numpy` - Array operations
- `scipy` - Statistical functions and distributions
- `astropy` - Specialized time series analysis tools
- `lmfit` - Model fitting capabilities

When adding new dependencies, add them to `pyproject.toml` under `[project]` dependencies.
