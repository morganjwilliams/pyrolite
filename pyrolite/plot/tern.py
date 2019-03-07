import matplotlib.pyplot as plt
import numpy as np
import ternary as pyternary
import logging
from ..util.plot import (
    ABC_to_tern_xy,
    __DEFAULT_CONT_COLORMAP__,
    __DEFAULT_DISC_COLORMAP__,
)
from ..util.meta import get_additional_params
from ..comp.codata import close

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def ternary(
    arr,
    ax=None,
    color="0.5",
    alpha=1.0,
    marker="D",
    label=None,
    clockwise=True,
    scale=100.0,
    gridsize=10.0,
    figsize=(8.0, 4 * 3 ** 0.5),
    **kwargs
):
    """
    Ternary scatter diagrams. This function uses the :mod:`python-ternary` library
    (`gh.com/marcharper/python-ternary <https://github.com/marcharper/python-ternary>`).
    Additional keyword arguments are passed to :mod:`matplotlib` (see Other Parameters, below).

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        Array from which to draw data.
    ax : :class:`matplotlib.axes.Axes`, `None`
        The subplot to draw on.

    color : :class:`str` | :class:`list` | :class:`numpy.ndarray`
        Individual color or collection of :mod:`~matplotlib.colors` to be passed to matplotlib.
    alpha : :class:`float`, 1.
        Opacity for the plotted series.
    marker : :class:`str`, 'D'
        Matplotlib :mod:`~matplotlib.markers` designation.
    label : :class:`str`, `None`
        Label for the individual series.
    clockwise : :class:`bool`
        Whether to use a clockwise (True) or anticlockwise handedness for axes.
    scale : :class:`int`, 100.0
        Scale to be passed to python-ternary.
    gridsize : :class:`int`, 10.0
        Interval between ternary gridlines.
    figsize : :class:`tuple`
        Size of the figure to be generated, if not using an existing :class:`~matplotlib.axes.Axes`.

    {otherparams}

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the spiderplot is plotted.

    Todo
    -------
        * Create Ternary class for separate scatter, plot methods; layer of abstraction
        * Changing `clockwise` can render the plot invalid. Update to fix.

    See Also
    ---------
    :func:`matplotlib.pyplot.plot`
    :func:`matplotlib.pyplot.scatter`
    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    d1 = ax.__dict__.copy()

    # Checking if there's already a ternary axis
    tax = getattr(ax, "tax", None)
    if tax is None:
        fig, tax = pyternary.figure(ax=ax, scale=scale)

    # Set attribute for future reference
    ax.tax = tax
    points = close(arr) * scale

    valid_rows = np.isfinite(points).all(axis=-1)
    if valid_rows.any() and points.size:
        if points.ndim == 1:
            points = np.array([points])

        assert points.shape[1] == 3

        tax.scatter(points, c=color, marker=marker, alpha=alpha, label=label)

    if label is not None:
        tax.legend(frameon=False)

    # Check if there's already labels
    if not len(tax._labels.keys()):
        # python-ternary uses "right, top, left"
        tax.gridlines(multiple=gridsize, color="k", alpha=0.5)
        tax.ticks(linewidth=1, clockwise=clockwise, multiple=gridsize)
        tax.boundary(linewidth=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    return ax


_add_additional_parameters = True

ternary.__doc__ = ternary.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            ternary,
            plt.scatter,
            plt.plot,
            header="Other Parameters",
            indent=4,
            subsections=True,
        ),
    ][_add_additional_parameters]
)
