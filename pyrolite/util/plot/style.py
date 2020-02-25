"""
Functions for automated plot styling and argument handling.

Attributes
----------
DEFAULT_CONT_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default continuous colormap.
DEFAULT_DISC_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default discrete colormap.
"""
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.axes
import matplotlib.collections
import matplotlib.patches
from ..meta import subkwargs
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

DEFAULT_CONT_COLORMAP = plt.cm.viridis
DEFAULT_DISC_COLORMAP = plt.cm.tab10


def linekwargs(kwargs):
    """
    Get a subset of keyword arguments to pass to a matplotlib line-plot call.

    Parameters
    -----------
    kwargs : :class:`dict`
        Dictionary of keyword arguments to subset.

    Returns
    --------
    :class:`dict`
    """
    kw = subkwargs(
        kwargs,
        plt.plot,
        matplotlib.axes.Axes.plot,
        matplotlib.lines.Line2D,
        matplotlib.collections.Collection,
    )
    # could trim cmap and norm here, in case they get passed accidentally
    kw.update(
        **dict(alpha=kwargs.get("alpha"), label=kwargs.get("label"))
    )  # issues with introspection for alpha
    return kw


def scatterkwargs(kwargs):
    """
    Get a subset of keyword arguments to pass to a matplotlib scatter call.

    Parameters
    -----------
    kwargs : :class:`dict`
        Dictionary of keyword arguments to subset.

    Returns
    --------
    :class:`dict`
    """
    kw = subkwargs(
        kwargs,
        plt.scatter,
        matplotlib.axes.Axes.scatter,
        matplotlib.collections.Collection,
    )
    kw.update(
        **dict(alpha=kwargs.get("alpha"), label=kwargs.get("label"))
    )  # issues with introspection for alpha
    return kw


def patchkwargs(kwargs):
    kw = subkwargs(
        kwargs,
        matplotlib.axes.Axes.fill_between,
        matplotlib.collections.PolyCollection,
        matplotlib.patches.Patch,
    )
    kw.update(
        **dict(alpha=kwargs.get("alpha"), label=kwargs.get("label"))
    )  # issues with introspection for alpha
    return kw


def _mpl_sp_kw_split(kwargs):
    """
    Process keyword arguments supplied to a matplotlib plot function.

    Returns
    --------
    :class:`tuple` ( :class:`dict`, :class:`dict` )
    """
    sctr_kwargs = scatterkwargs(kwargs)
    # c kwarg is first priority, if it isn't present, use the color arg
    if sctr_kwargs.get("c") is None:
        sctr_kwargs = {**sctr_kwargs, **{"c": kwargs.get("color")}}

    line_kwargs = linekwargs(kwargs)
    return sctr_kwargs, line_kwargs


def marker_cycle(markers=["D", "s", "o", "+", "*"]):
    """
    Cycle through a set of markers.

    Parameters
    -----------
    markers : :class:`list`
        List of markers to provide to matplotlib.
    """
    return itertools.cycle(markers)


def mappable_from_values(values, cmap=DEFAULT_CONT_COLORMAP, **kwargs):
    """
    Create a scalar mappable object from an array of values.

    Returns
    -----------
    :class:`matplotlib.cm.ScalarMappable`
    """
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(values)
    return sm
