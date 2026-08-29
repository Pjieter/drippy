# DRIPPY

Discover, Refine, Inspect, Present in Python, using EDA principles outlined by NIST/SEMATECH e-Handbook of Statistical Methods, ([this link](http://www.itl.nist.gov/div898/handbook/eda/eda.htm)), to analyse data.

[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/Pjieter/drippy)
[![Documentation Status](https://readthedocs.org/projects/drippy/badge/?version=latest)](https://drippy.readthedocs.io/en/latest/?badge=latest)
[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=Pjieter_drippy&metric=alert_status)](https://sonarcloud.io/dashboard?id=Pjieter_drippy)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=Pjieter_drippy&metric=coverage)](https://sonarcloud.io/dashboard?id=Pjieter_drippy)
[![build](https://github.com/Pjieter/drippy/actions/workflows/build.yml/badge.svg)](https://github.com/Pjieter/drippy/actions/workflows/build.yml)
[![cffconvert](https://github.com/Pjieter/drippy/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/Pjieter/drippy/actions/workflows/cffconvert.yml)
[![sonarcloud](https://github.com/Pjieter/drippy/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/Pjieter/drippy/actions/workflows/sonarcloud.yml)
[![link-check](https://github.com/Pjieter/drippy/actions/workflows/link-check.yml/badge.svg)](https://github.com/Pjieter/drippy/actions/workflows/link-check.yml)

## How to use drippy

drippy revolves around `EDAData`, a validated container for your data that
exposes every plot as a fluent method. The same plots are also available as
standalone functions that accept an `EDAData` instance as their first
argument.

```python
import numpy as np

from drippy import EDAData
from drippy import histogram

rng = np.random.default_rng(42)
y = rng.normal(loc=688.0, scale=65.0, size=200)
data = EDAData(y=y)

# Fluent API: NIST's recommended first stop for any EDA, the 4-plot
fig, axes = data.four_plot()

# Standalone function: same result, explicit function call
fig, ax = histogram(data, bins=20)
```

`EDAData` accepts optional `t` (time series), `x` (one-factor), or `factors`
(multi-factor/DOE) arguments to unlock the corresponding plot families:

- **Univariate** (`y = c + e`): four-plot, histogram, run-sequence plot, lag
  plot, bootstrap plot, box-cox normality/linearity plots, normal probability
  plot, probability plot, PPCC plot, QQ plot, Weibull plot
- **Time series** (`y = f(t) + e`): autocorrelation, spectral, and complex
  demodulation plots
- **One-factor** (`y = f(x) + e`): box plot, scatter plot, mean/sd plots,
  bihistogram
- **Multi-factor / DOE**: DOE mean/sd/scatter plots, contour plot
- **Regression**: six-plot, linear slope/intercept/correlation/residual-sd
  plots
- **Comparative**: block plot, star plot, Youden plot

## Installation

Install drippy from PyPI (distribution name `drippy-eda`; the import name is `drippy`):

```console
pip install drippy-eda
```

### Development installation

To work on drippy itself, clone the repository and install it with
[uv](https://docs.astral.sh/uv/):

```console
git clone git@github.com:Pjieter/drippy.git && cd drippy && uv sync --all-groups
```

## Documentation

The documentation can be found here: [https://drippy.readthedocs.io/en/latest/](https://drippy.readthedocs.io/en/latest/)
## Contributing

If you want to contribute to the development of drippy,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).


