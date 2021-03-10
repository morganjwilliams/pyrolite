import matplotlib.pyplot as plt
import numpy as np
from ..util.plot.axes import init_axes
from ..util.plot.style import linekwargs, scatterkwargs
from ..util.log import Handle

logger = Handle(__name__)



def stem(x, y, ax=None, orientation="horizontal", **kwargs):
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

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the stem diagram is plotted.
    """
    ax = init_axes(ax=ax, **kwargs)

    orientation = orientation.lower()
    xs, ys = np.array([x, x]), np.array([np.zeros_like(y), y])
    positivey = (y > 0 | ~np.isfinite(y)).all() | np.allclose(y, 0)
    kwargs["color"] = kwargs.get("color") or "0.5"
    if "h" in orientation:
        ax.plot(xs, ys, **linekwargs(kwargs))
        ax.scatter(x, y, **scatterkwargs(kwargs))
        if positivey:
            ax.set_ylim(0, ax.get_ylim()[1])
    else:
        ax.plot(ys, xs, **linekwargs(kwargs))
        ax.scatter(y, x, **scatterkwargs(kwargs))
        if positivey:
            ax.set_xlim(0, ax.get_xlim()[1])

    return ax
