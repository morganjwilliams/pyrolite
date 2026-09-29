"""
Functions for creating and modifying legend entries for matplotlib.

Todo
------

* Functions for working with and modifying legend entries.
  ax.lines + ax.patches + ax.collections + ax.containers, handle ax.parasites
"""

from copy import copy

import matplotlib.axes
import matplotlib.lines
import matplotlib.patches

from ..log import Handle

logger = Handle(__name__)


def proxy_rect(**kwargs) -> matplotlib.patches.Rectangle:
    """
    Generates a legend proxy for a filled region.

    Returns
    ----------
    matplotlib.patches.Rectangle
    """
    return matplotlib.patches.Rectangle((0, 0), 1, 1, **kwargs)


def proxy_line(**kwargs) -> matplotlib.lines.Line2D:
    """
    Generates a legend proxy for a line region.

    Returns
    ----------
    matplotlib.lines.Line2D
    """
    return matplotlib.lines.Line2D(range(1), range(1), **kwargs)


def modify_legend_handles(ax: matplotlib.axes.Axes, **kwargs) -> tuple[list, list[str]]:
    """
    Modify the handles of a legend based for a single axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis for which to obtain modifed legend handles.

    Returns
    -------
    handles : list
        Handles to be passed to a legend call.
    labels : list
        Labels to be passed to a legend call.
    """
    hndls, labls = ax.get_legend_handles_labels()
    _hndls = []
    for h in hndls:
        _h = copy(h)
        _h.update(kwargs)
        _hndls.append(_h)
    return _hndls, labls
