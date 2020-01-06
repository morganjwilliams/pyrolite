import matplotlib.pyplot as plt
import numpy as np
import logging
from ..util.meta import subkwargs
from ..util.plot import init_axes

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def stem(x, y, ax=None, orientation="horizontal", color="0.5", **kwargs):
    """
    Create a stem (or 'lollipop') plot, with optional orientation.

    Parameters
    -----------
    x, y : :class:`numpy.ndarray`
        1D arrays for independent and dependent axes.
    ax : :class:`matplotlib.axes.Axes`, :code:`None`
        The subplot to draw on.
    orientation : :class:`str`
        Orientation of the plot (horizontal or vertical).
    color : :class:`str`
        Color of lines and markers (unless otherwise overridden).

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the stem diagram is plotted.
    """
    ax = init_axes(ax=ax, **kwargs)

    orientation = orientation.lower()
    xs, ys = [x, x], [np.zeros_like(y), y]
    positivey = (y > 0 | ~np.isfinite(y)).all() | np.allclose(y, 0)
    if "h" in orientation:
        ax.plot(xs, ys, color=color, **subkwargs(kwargs, ax.plot))
        ax.scatter(x, y, **{"c": color, **subkwargs(kwargs, ax.scatter)})
        if positivey:
            ax.set_ylim(0, ax.get_ylim()[1])
    else:
        ax.plot(ys, xs, color=color, **subkwargs(kwargs, ax.plot))
        ax.scatter(y, x, **{"c": color, **subkwargs(kwargs, ax.scatter)})
        if positivey:
            ax.set_xlim(0, ax.get_xlim()[1])

    return ax
