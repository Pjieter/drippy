# `drippy` developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

drippy uses [uv](https://docs.astral.sh/uv/) for everything: virtual
environments, dependency groups, builds, and running tools. Install uv, then
from the project root:

```shell
# create .venv, install drippy as editable, plus the dev and docs groups
uv sync --all-groups
```

Dependency groups (`dev`, `docs`) are declared under `[dependency-groups]` in
`pyproject.toml`. To install only the docs tooling:

```shell
uv sync --group docs
```

Prefix commands with `uv run` so they use the project environment; no manual
activation needed.

## Running the tests

```shell
uv run pytest -v          # all tests
uv run pytest -n auto     # in parallel (pytest-xdist)
```

To test against every supported Python version in isolated environments:

```shell
uv run tox
```

### Test coverage

```shell
uv run coverage run
uv run coverage report
```

`coverage` can also generate output in HTML and other formats; see
`uv run coverage help` for more information.

## Running linters locally

Linting, import sorting, and formatting are all done with
[ruff](https://docs.astral.sh/ruff/):

```shell
uv run ruff check .          # lint
uv run ruff check . --fix    # lint with automatic fixing
uv run ruff format           # format
```

You can run ruff automatically on every commit by enabling the git hook in
`.githooks/pre-commit`:

```shell
git config --local core.hooksPath .githooks
```

## Generating the API docs

```shell
cd docs
uv run sphinx-build -b html . _build/html
```

The documentation will be in `docs/_build/html`. The example notebooks under
`docs/examples/` are executed during the build (myst-nb, cached), so a broken
example fails the build.

To find undocumented Python objects run

```shell
cd docs
uv run sphinx-build -b coverage . _build/coverage
cat _build/coverage/python.txt
```

To [test snippets](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html)
in documentation run

```shell
cd docs
uv run sphinx-build -b doctest . _build/doctest
```

## Versioning

Bumping the version across all files is done with
[bump-my-version](https://github.com/callowayproject/bump-my-version), e.g.

```shell
uv run bump-my-version bump major  # bumps from e.g. 0.3.2 to 1.0.0
uv run bump-my-version bump minor  # bumps from e.g. 0.3.2 to 0.4.0
uv run bump-my-version bump patch  # bumps from e.g. 0.3.2 to 0.3.3
```

This updates `pyproject.toml`, `src/drippy/__init__.py`, `CITATION.cff`, and
`docs/conf.py`. It does not touch `date-released` in `CITATION.cff` or the
dates in `CHANGELOG.md`; set those by hand.

## Making a release

Releases are published to PyPI automatically by the
[`release.yml`](.github/workflows/release.yml) workflow when a GitHub release
is published. The full procedure, including the one-time trusted-publisher
setup on pypi.org, is in
[CONTRIBUTING.md](CONTRIBUTING.md#you-want-to-make-a-new-release-of-the-code-base).

To check the distribution locally before tagging:

```shell
uv build
uvx twine check dist/*
```
