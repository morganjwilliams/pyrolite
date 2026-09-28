import matplotlib.axes
import numpy as np
import pandas as pd

from ..util.log import Handle
from ..util.plot.axes import init_axes
from ..util.plot.style import linekwargs, scatterkwargs

logger = Handle(__name__)


def stem(
    x: np.ndarray[tuple[int], np.dtype[np.number]] | pd.Series,
    y: np.ndarray[tuple[int], np.dtype[np.number]] | pd.Series,
    ax: matplotlib.axes.Axes | None = None,
    orientation: str = "horizontal",
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Create a stem (or 'lollipop') plot, with optional orientation.

    Parameters
    -----------
    x, y : numpy.ndarray
        1D arrays for independent and dependent axes.
    ax : matplotlib.axes.Axes
        The subplot to draw on.
    orientation : str
        Orientation of the plot (horizontal or vertical).

    Returns
    -------
    matplotlib.axes.Axes
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
