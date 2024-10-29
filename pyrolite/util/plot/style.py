"""
Functions for automated plot styling and argument handling.

Attributes
----------
DEFAULT_CONT_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default continuous colormap.
DEFAULT_DISC_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default discrete colormap.
"""

import itertools
from pathlib import Path

import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

from ...comp.codata import close
from ..general import copy_file
from ..log import Handle
from ..meta import pyrolite_datafolder, subkwargs
from .helpers import get_centroid
from .transform import xy_to_tlr

logger = Handle(__name__)

DEFAULT_CONT_COLORMAP = plt.cm.viridis
DEFAULT_DISC_COLORMAP = plt.cm.tab10


def _export_mplstyle(
    src=pyrolite_datafolder("_config") / "pyrolite.mplstyle", refresh=False
):
    """
    Export a matplotlib style file to the matplotlib style library such that one can
    use e.g. `matplotlib.style.use("pyrolite")`.

    Parameters
    -----------
    src : :class:`str` | :class:`pathlib.Path`
        File path for the style file to be exported.
    refresh : :class:`bool`
        Whether to re-export a style file (e.g. after updating) even if it
        already exists in the matplotlib style libary.
    """
    src_fn = Path(src)
    dest_dir = Path(matplotlib.get_configdir()) / "stylelib"
    dest_fn = dest_dir / src.name
    if (not dest_fn.exists()) or refresh:
        logger.debug("Exporting pyrolite.mplstyle to matplotlib config folder.")
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True)
        copy_file(src_fn, dest_dir)  # copy to the destination DIR
        logger.debug("Reloading matplotlib")
    matplotlib.style.reload_library()  # needed to load in pyrolite style NOW


def _restyle(f, **_style):
    """
    A decorator to set the default keyword arguments for :mod:`matplotlib`
    functions and classes which are not contained in the `matplotlibrc` file.
    """

    def wrapped(*args, **kwargs):
        return f(*args, **{**_style, **kwargs})

    wrapped.__name__ = f.__name__
    wrapped.__doc__ = f.__doc__
    return wrapped


def _export_nonRCstyles(**kwargs):
    """
    Export default options for parameters not in rcParams using :func:`_restyle`.
    """
    matplotlib.axes.Axes.legend = _restyle(
        matplotlib.axes.Axes.legend, **{"bbox_to_anchor": (1, 1), **kwargs}
    )
    matplotlib.figure.Figure.legend = _restyle(
        matplotlib.figure.Figure.legend, bbox_to_anchor=(1, 1)
    )


_export_mplstyle()
_export_nonRCstyles(handler_map={tuple: HandlerTuple(ndivide=None)})
matplotlib.style.use("pyrolite")


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
        **dict(
            alpha=kwargs.get("alpha"),
            label=kwargs.get("label"),
            clip_on=kwargs.get("clip_on", True),
        )
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
        **dict(
            alpha=kwargs.get("alpha"),
            label=kwargs.get("label"),
            clip_on=kwargs.get("clip_on", True),
        )
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
        **dict(
            alpha=kwargs.get("alpha"),
            label=kwargs.get("label"),
            clip_on=kwargs.get("clip_on", True),
        )
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
    ----------
    markers : :class:`list`
        List of markers to provide to matplotlib.
    """
    return itertools.cycle(markers)


def mappable_from_values(values, cmap=DEFAULT_CONT_COLORMAP, norm=None, **kwargs):
    """
    Create a scalar mappable object from an array of values.

    Returns
    -------
    :class:`matplotlib.cm.ScalarMappable`
    """
    if isinstance(values, pd.Series):
        values = values.values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(values[np.isfinite(values)])
    return sm


def ternary_color(
    tlr,
    alpha=1.0,
    colors=([1, 0, 0], [0, 1, 0], [0, 0, 1]),
    coefficients=(1, 1, 1),
):
    """
    Color a set of points by their ternary combinations of three specified colors.

    Parameters
    ----------
    tlr : :class:`numpy.ndarray`

    alpha : :class:`float`
        Alpha coefficient for the colors; to be applied *multiplicatively* with
        any existing alpha value (for RGBA colors specified).
    colors : :class:`tuple`
        Set of colours corresponding to the top, left and right verticies,
        respectively.
    coefficients : :class:`tuple`
        Coefficients for the ternary data to adjust the centre.

    Returns
    -------
    :class:`numpy.ndarray`
        Color array for the ternary points.
    """
    colors = np.array([matplotlib.colors.to_rgba(c) for c in colors], dtype=float)
    _tlr = close(np.array(tlr) * np.array(coefficients))
    color = np.atleast_2d(_tlr @ colors)
    color[:, -1] *= alpha * (1 - 10e-7)  # avoid 'greater than 1' issues
    return color


def color_ternary_polygons_by_centroid(
    ax=None,
    patches=None,
    alpha=1.0,
    colors=([1, 0, 0], [0, 1, 0], [0, 0, 1]),
    coefficients=(1, 1, 1),
):
    """
    Color a set of polygons within a ternary diagram by their centroid colors.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Ternary axes to check for patches, if patches is not supplied.
    patches : :class:`list`
        List of ternary-hosted patches to apply color to.
    alpha : :class:`float`
        Alpha coefficient for the colors; to be applied *multiplicatively* with
        any existing alpha value (for RGBA colors specified).
    colors : :class:`tuple`
        Set of colours corresponding to the top, left and right verticies,
        respectively.
    coefficients : :class:`tuple`
        Coefficients for the ternary data to adjust the centre.

    Returns
    -------
    patches : :class:`list`
        List of patches, with updated facecolors.
    """

    if patches is None:
        if ax is None:
            raise NotImplementedError("Either an axis or patches need to be supplied.")
        patches = ax.patches

    for poly in patches:
        xy = get_centroid(poly)
        tlr = xy_to_tlr(np.array([xy]))[0]
        color = ternary_color(
            tlr, alpha=alpha, colors=colors, coefficients=coefficients
        )
        poly.set_facecolor(color)

    return patches
