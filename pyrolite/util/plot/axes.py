"""
Functions for creating, ordering and modifying :class:`~matplolib.axes.Axes`.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..log import Handle
from ..meta import subkwargs

logger = Handle(__name__)


def get_ordered_axes(fig):
    """
    Get the axes from a figure, which may or may not have been modified by
    pyrolite functions. This ensures that ordering is preserved.
    """
    if hasattr(fig, "orderedaxes"):  # previously modified
        axes = fig.orderedaxes
    else:  # unmodified axes
        axes = fig.axes
    return axes


def get_axes_index(ax):
    """
    Get the three-digit integer index of a subplot in a regular grid.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Axis to to get the gridspec index for.

    Returns
    -----------
    :class:`tuple`
        Rows, columns and axis index for the gridspec.
    """
    nrow, ncol = ax.get_gridspec()._nrows, ax.get_gridspec()._ncols
    index = get_ordered_axes(ax.figure).index(ax)
    triple = nrow, ncol, index + 1
    return triple


def replace_with_ternary_axis(ax):
    """
    Replace a specified axis with a ternary equivalent.

    Parameters
    ------------
    ax : :class:`~matplotlib.axes.Axes`

    Returns
    ------------
    tax : :class:`~mpltern.ternary.TernaryAxes`
    """
    if ax.name != "ternary":
        if not check_default_axes(ax):
            if not check_empty(ax):
                warnings.warn(
                    "Non-empty, non-default bivariate axes being replaced with ternary axes."
                )
            else:
                logger.info(
                    "Non-default bivraite axes being replaced with ternary axes."
                )
    fig = ax.figure
    axes = get_ordered_axes(fig)
    idx = axes.index(ax)
    tax = fig.add_subplot(*get_axes_index(ax), projection="ternary")
    fig.add_axes(tax)  # make sure the axis is added to fig.children
    fig.delaxes(ax)  # remove the original axes
    # update figure ordered axes
    fig.orderedaxes = [a if ix != idx else tax for (ix, a) in enumerate(axes)]
    return tax


def label_axes(ax, labels=[], **kwargs):
    """
    Convenience function for labelling rectilinear and ternary axes.

    Parameters
    -----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to label.
    labels : :class:`list`
        List of labels: [x, y] | or [t, l, r]
    """
    if (ax.name == "ternary") and (len(labels) == 3):
        tvar, lvar, rvar = labels
        ax.set_tlabel(tvar, **kwargs)
        ax.set_llabel(lvar, **kwargs)
        ax.set_rlabel(rvar, **kwargs)
    elif len(labels) == 2:
        xvar, yvar = labels
        ax.set_xlabel(xvar, **kwargs)
        ax.set_ylabel(yvar, **kwargs)
    else:
        raise NotImplementedError


def axes_to_ternary(ax):
    """
    Set axes to ternary projection after axis creation. As currently implemented,
    note that this will replace and reorder axes as acecessed from the figure (the
    ternary axis will then be at the end), and as such this returns a list of axes
    in the correct order.

    Parameters
    -----------
    ax : :class:`~matplotlib.axes.Axes` | :class:`list` (:class:`~matplotlib.axes.Axes`)
        Axis (or axes) to convert projection for.

    Returns
    ---------
    axes : :class:`list' (:class:`~matplotlib.axes.Axes`, class:`~mpltern.ternary.TernaryAxes`)
    """

    if isinstance(ax, (list, np.ndarray, tuple)):  # multiple Axes specified
        fig = ax[0].figure
        for a in ax:  # axes to set to ternary
            replace_with_ternary_axis(a)
    else:  # a single Axes is passed
        fig = ax.figure
        replace_with_ternary_axis(ax)
    return fig.orderedaxes


def check_default_axes(ax):
    """
    Simple test to check whether an axis is empty of artists and hasn't been
    rescaled from the default extent.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to check for artists and scaling.

    Returns
    -------
    :class:`bool`
    """

    if np.allclose(ax.axis(), np.array([0, 1, 0, 1])):
        return check_empty(ax)
    else:
        return False


def check_empty(ax):
    """
    Simple test to check whether an axis is empty of artists.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to check for artists.

    Returns
    -------
    :class:`bool`
    """
    if not (ax.lines + ax.collections + ax.patches + ax.artists + ax.texts + ax.images):
        return True
    else:
        return False


def init_axes(ax=None, projection=None, minsize=1.0, **kwargs):
    """
    Get or create an Axes from an optionally-specified starting Axes.

    Parameters
    -----------
    ax : :class:`~matplotlib.axes.Axes`
        Specified starting axes, optional.
    projection : :class:`str`
        Whether to create a projected (e.g. ternary) axes.
    minsize : :class:`float`
        Minimum figure dimension (inches).

    Returns
    --------
    ax : :class:`~matplotlib.axes.Axes`
    """
    if "figsize" in kwargs.keys():
        fs = kwargs["figsize"]
        kwargs["figsize"] = (
            max(fs[0], minsize),
            max(fs[1], minsize),
        )  # minimum figsize
    if projection is not None:  # e.g. ternary
        if ax is None:
            fig, ax = plt.subplots(
                1,
                subplot_kw=dict(projection=projection),
                **subkwargs(kwargs, plt.subplots, plt.figure),
            )
        else:  # axes passed
            if ax.name != "ternary":
                # if an axis is converted previously, but the original axes reference
                # is used again, we'll end up with an error
                current_axes = get_ordered_axes(ax.figure)
                try:
                    ix = current_axes.index(ax)
                    axes = axes_to_ternary(ax)  # returns list of axes
                    ax = axes[ix]
                except ValueError:  # ax is not in list
                    # ASSUMPTION due to mis-referencing:
                    # take the first ternary one
                    ax = [a for a in current_axes if a.name == "ternary"][0]
            else:
                pass
    else:
        if ax is None:
            fig, ax = plt.subplots(1, **subkwargs(kwargs, plt.subplots, plt.figure))
    return ax


def share_axes(axes, which="xy"):
    """
    Link the x, y or both axes across a group of :class:`~matplotlib.axes.Axes`.

    Parameters
    -----------
    axes : :class:`list`
        List of axes to link.
    which : :class:`str`
        Which axes to link. If :code:`x`, link the x-axes; if :code:`y` link the y-axes,
        otherwise link both.
    """
    if which == "both":
        which = "xy"
    if "x" in which:
        [a.sharex(axes[0]) for a in axes[1:]]
    if "y" in which:
        [a.sharey(axes[0]) for a in axes[1:]]


def get_twins(ax, which="y"):
    """
    Get twin axes of a specified axis.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to get twins for.
    which : :class:`str`
        Which twins to get (shared :code:`'x'`, shared :code:`'y'` or the concatenatation
        of both, :code:`'xy'`).

    Returns
    --------
    :class:`list`

    Notes
    ------
    This function was designed to assist in avoiding creating a series of duplicate
    axes when replotting on an existing axis using a function which would typically
    create a twin axis.
    """
    s = []
    if "y" in which:
        s += ax.get_shared_y_axes().get_siblings(ax)
    if "x" in which:
        s += ax.get_shared_x_axes().get_siblings(ax)
    return list(
        set([a for a in s if (a is not ax) & (a.bbox.bounds == ax.bbox.bounds)])
    )


def subaxes(ax, side="bottom", width=0.2, moveticks=True):
    """
    Append a sub-axes to one side of an axes.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to append a sub-axes to.
    side : :class:`str`
        Which side to append the axes on.
    width : :class:`float`
        Fraction of width to give to the subaxes.
    moveticks : :class:`bool`
        Whether to move ticks to the outer axes.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Subaxes instance.
    """
    div = make_axes_locatable(ax)
    ax.divider = div

    if side in ["bottom", "top"]:
        which = "x"
        subax = div.append_axes(side, width, pad=0, sharex=ax)
        div.subax = subax
        subax.yaxis.set_visible(False)
        subax.spines["left"].set_visible(False)
        subax.spines["right"].set_visible(False)

    else:
        which = "y"
        subax = div.append_axes(side, width, pad=0, sharex=ax)
        div.subax = subax
        subax.yaxis.set_visible(False)
        subax.spines["top"].set_visible(False)
        subax.spines["bottom"].set_visible(False)

    share_axes([ax, subax], which=which)
    if moveticks:
        ax.tick_params(
            axis=which, which="both", bottom=False, top=False, labelbottom=False
        )
    return subax


def add_colorbar(mappable, **kwargs):
    """
    Adds a colorbar to a given mappable object.

    Source: http://joseph-long.com/writing/colorbars/

    Parameters
    ----------
    mappable
        The Image, ContourSet, etc. to which the colorbar applies.

    Returns
    -------
    :class:`matplotlib.colorbar.Colorbar`

    Todo
    ----
    *  Where no mappable specificed, get most recent axes, and check for collections etc
    """
    ax = kwargs.get("ax", None)
    if hasattr(mappable, "axes"):
        ax = ax or mappable.axes
    elif hasattr(mappable, "ax"):
        ax = ax or mappable.ax

    position = kwargs.pop("position", "right")
    size = kwargs.pop("size", "5%")
    pad = kwargs.pop("pad", 0.05)

    fig = ax.figure
    if ax.name == "ternary":
        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
        colorbar = fig.colorbar(mappable, cax=cax, **kwargs)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position, size=size, pad=pad)
        colorbar = fig.colorbar(mappable, cax=cax, **kwargs)
    return colorbar
