import matplotlib.pyplot as plt
import numpy as np
import ternary as pyternary
import logging
from ..util.plot import (
    ABC_to_xy,
    __DEFAULT_CONT_COLORMAP__,
    __DEFAULT_DISC_COLORMAP__,
    ternary_patch,
)
from ..util.meta import get_additional_params, subkwargs
from ..comp.codata import close

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def ternary(
    arr,
    ax=None,
    color=None,
    alpha=1.0,
    marker="D",
    label=None,
    clockwise=True,
    scale=100.0,
    gridsize=20.0,
    figsize=(8.0, 4 * 3 ** 0.5),
    no_ticks=False,
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
        Size of the figure to be generated, if not using an existing
        :class:`~matplotlib.axes.Axes`.
    no_ticks : :class:`bool`
        Whether to suppress ticks and tick labels.

    {otherparams}

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the spiderplot is plotted.

    Todo
    -------
        * Create Ternary class for separate scatter, plot methods; layer of abstraction
        * Changing `clockwise` can render the plot invalid. Update to fix.

    Notes
    -------
        * To create unfilled markers, pass :code:`edgecolors=<color>, c="none"`
        * To edit marker edgewiths, pass :code:`linewidths=<width>`

    .. seealso::

        Functions:

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

    if not hasattr(tax, "patch"):
        tax.patch = ternary_patch(
            scale=scale,
            color=ax.patch.get_facecolor(),
            yscale=np.sqrt(3) / 2,
            zorder=-10,
        )
        ax.add_artist(tax.patch)
    # Set attribute for future reference
    ax.tax = tax
    points = close(arr) * scale

    valid_rows = np.isfinite(points).all(axis=-1)
    if valid_rows.any() and points.size:
        if points.ndim == 1:
            points = np.array([points])

        assert points.shape[1] == 3
        config = dict(c=color, marker=marker, alpha=alpha, label=label)
        if isinstance(color, (str, tuple)):
            config["color"] = config.pop("c")

        config = {**config, **subkwargs(kwargs, ax.scatter)}
        if ("cmap" in config) and ("c" in config):
            vmin, vmax = np.nanmin(config["c"]), np.nanmax(config["c"])
            config["norm"] = config.get("norm", plt.Normalize(vmin=vmin, vmax=vmax))
        if "norm" in config:
            config["vmin"] = config["norm"].vmin
            config["vmax"] = config["norm"].vmax
        tax.scatter(points, **config)

    if label is not None:
        tax.legend(frameon=False)

    # Check if there's already labels
    fontsize = kwargs.get("fontsize", 8.0)
    if not len(tax._labels.keys()):
        # python-ternary uses "right, top, left"
        tax.gridlines(multiple=gridsize, color="k", alpha=0.5)
        if not no_ticks:
            tax.ticks(
                linewidth=1,
                clockwise=clockwise,
                multiple=gridsize,
                fontsize=fontsize,
                offset=0.03,
            )
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
