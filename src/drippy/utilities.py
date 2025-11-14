import numpy as np


def bl_filt(y, half_width):
    """
    Simple Blackman filter.

    The end effects are handled by calculating the weighted
    average of however many points are available, rather than
    by zero-padding.
    """
    nf = half_width * 2 + 1
    x = np.linspace(-1, 1, nf, endpoint=True)
    x = x[1:-1]  # chop off the useless endpoints with zero weight
    w = 0.42 + 0.5 * np.cos(x * np.pi) + 0.08 * np.cos(x * 2 * np.pi)
    ytop = np.convolve(y, w, mode="same")
    ybot = np.convolve(np.ones_like(y), w, mode="same")

    return ytop / ybot
