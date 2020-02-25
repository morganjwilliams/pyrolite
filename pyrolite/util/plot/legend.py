"""
Functions for creating and modifying legend entries for matplotlib.
"""
import matplotlib.patches
import matplotlib.lines
from copy import copy
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def proxy_rect(**kwargs):
    """
    Generates a legend proxy for a filled region.

    Returns
    ----------
    :class:`matplotlib.patches.Rectangle`
    """
    return matplotlib.patches.Rectangle((0, 0), 1, 1, **kwargs)


def proxy_line(**kwargs):
    """
    Generates a legend proxy for a line region.

    Returns
    ----------
    :class:`matplotlib.lines.Line2D`
    """
    return matplotlib.lines.Line2D(range(1), range(1), **kwargs)


def modify_legend_handles(ax, **kwargs):
    """
    Modify the handles of a legend based for a single axis.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis for which to obtain modifed legend handles.

    Returns
    -------
    handles : :class:`list`
        Handles to be passed to a legend call.
    labels : :class:`list`
        Labels to be passed to a legend call.
    
    """
    hndls, labls = ax.get_legend_handles_labels()
    _hndls = []
    for h in hndls:
        _h = copy(h)
        _h.update(kwargs)
        _hndls.append(_h)
    return _hndls, labls
