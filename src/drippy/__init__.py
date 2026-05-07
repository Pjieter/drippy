"""drippy — EDA plotting library following NIST/SEMATECH principles."""

import logging
from drippy.data import EDAData
from drippy.onefactor import bihistogram
from drippy.onefactor import box_plot
from drippy.onefactor import mean_plot
from drippy.onefactor import qq_plot
from drippy.onefactor import scatter_plot
from drippy.onefactor import sd_plot
from drippy.timeseries import autocorrelation_plot
from drippy.timeseries import complex_demodulation_amplitude_plot
from drippy.timeseries import complex_demodulation_phase_plot
from drippy.timeseries import spectral_plot
from drippy.univariate import bootstrap_plot
from drippy.univariate import box_cox_linearity_plot
from drippy.univariate import box_cox_normality_plot
from drippy.univariate import four_plot
from drippy.univariate import histogram
from drippy.univariate import lag_plot
from drippy.univariate import normal_probability_plot
from drippy.univariate import ppcc_plot
from drippy.univariate import probability_plot
from drippy.univariate import run_sequence_plot
from drippy.univariate import weibull_plot

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Michiel Dubbelman"
__email__ = "m.p.dubbelman@tudelft.nl"
__version__ = "0.1.0"

__all__ = [
    "EDAData",
    "autocorrelation_plot",
    "bihistogram",
    "bootstrap_plot",
    "box_cox_linearity_plot",
    "box_cox_normality_plot",
    "box_plot",
    "complex_demodulation_amplitude_plot",
    "complex_demodulation_phase_plot",
    "four_plot",
    "histogram",
    "lag_plot",
    "mean_plot",
    "normal_probability_plot",
    "ppcc_plot",
    "probability_plot",
    "qq_plot",
    "run_sequence_plot",
    "scatter_plot",
    "sd_plot",
    "spectral_plot",
    "weibull_plot",
]
