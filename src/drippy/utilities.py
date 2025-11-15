import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def get_figure_and_axes(
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Get or create figure and axes for plotting.

    Args:
        fig (Figure, optional): Existing Figure or None to create new. Defaults to None.
        ax (Axes, optional): Existing Axes or None to create new. Defaults to None.

    Returns:
        tuple[Figure, Axes]: Figure and Axes objects for plotting.
    """
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)
    return fig, ax


def bl_filt(y: np.ndarray, half_width: int) -> np.ndarray:
    """Simple Blackman filter.

    The end effects are handled by calculating the weighted
    average of however many points are available, rather than
    by zero-padding.

    Args:
        y (np.ndarray): Input signal to be filtered.
        half_width (int): Half-width of the filter window. Total window size will be 2*half_width + 1.

    Returns:
        np.ndarray: Filtered signal with the same shape as input.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array.")
    if not isinstance(half_width, int) or half_width < 1:
        raise ValueError(f"half_width must be a positive integer. Got {half_width}.")
    nf = half_width * 2 + 1
    x = np.linspace(-1, 1, nf, endpoint=True)
    x = x[1:-1]  # chop off the useless endpoints with zero weight
    w = 0.42 + 0.5 * np.cos(x * np.pi) + 0.08 * np.cos(x * 2 * np.pi)
    ytop = np.convolve(y, w, mode="same")
    ybot = np.convolve(np.ones_like(y), w, mode="same")

    return ytop / ybot
