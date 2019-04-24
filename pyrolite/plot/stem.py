import matplotlib.pyplot as plt
import numpy as np
import logging
from pyrolite.util.meta import subkwargs

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def stem(x, y, ax=None, orientation="horizontal", color="0.5", figsize=None, **kwargs):
    """
    Create a stem (or 'lollipop') plot, with optional orientation.
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    xs, ys = [x, x], [np.zeros_like(y), y]
    if "h" in orientation:
        ax.plot(xs, ys, color=color, **subkwargs(kwargs, ax.plot))
        ax.scatter(x, y, **{"c": color, **subkwargs(kwargs, ax.scatter)})
        if (y > 0).all() | np.allclose(y, 0):
            ax.set_ylim(0, ax.get_ylim()[1])
    else:
        ax.plot(ys, xs, color=color, **subkwargs(kwargs, ax.plot))
        ax.scatter(y, x, **{"c": color, **subkwargs(kwargs, ax.scatter)})
        if (y > 0).all() | np.allclose(y, 0):
            ax.set_xlim(0, ax.get_xlim()[1])

    return ax
