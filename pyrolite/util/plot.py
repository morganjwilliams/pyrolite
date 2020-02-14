"""
Utility functions for working with matplotlib.

Todo
------

    * Functions for working with and modifying legend entries.

        ax.lines + ax.patches + ax.collections + ax.containers, handle ax.parasites


Attributes
----------
DEFAULT_CONT_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default continuous colormap.
DEFAULT_DICS_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default discrete colormap.
USE_PCOLOR : :class:`bool`
    Option to use the :func:`matplotlib.pyplot.pcolor` function in place
    of :func:`matplotlib.pyplot.pcolormesh`.
"""
import os
import inspect
import itertools
from copy import copy
from types import MethodType
from pathlib import Path
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy.stats.kde import gaussian_kde
import scipy.spatial
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.path
import matplotlib.collections
import matplotlib.artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes
from matplotlib.transforms import Bbox
import mpltern.ternary
from ..util.math import (
    eigsorted,
    nancov,
    interpolate_line,
    flattengrid,
    linspc_,
    logspc_,
)
from ..util.distributions import sample_kde, sample_ternary_kde
from ..util.missing import cooccurence_pattern
from ..util.meta import subkwargs
from ..comp.codata import close, alr, ilr, clr, inverse_alr, inverse_clr, inverse_ilr
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

try:
    from sklearn.decomposition import PCA
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)

try:
    import statsmodels.api as sm

    HAVE_SM = True
except ImportError:
    HAVE_SM = False


DEFAULT_CONT_COLORMAP = plt.cm.viridis
DEFAULT_DISC_COLORMAP = plt.cm.tab10
USE_PCOLOR = False
FONTSIZE = 12


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


def marker_cycle(markers=["D", "s", "o", "+", "*"]):
    """
    Cycle through a set of markers.

    Parameters
    -----------
    markers : :class:`list`
        List of markers to provide to matplotlib.
    """
    return itertools.cycle(markers)


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
        tax = replace_with_ternary_axis(ax)
    return fig.orderedaxes


def init_axes(ax=None, projection=None, **kwargs):
    """
    Get or create an Axes from an optionally-specified starting Axes.

    Parameters
    -----------
    ax : :class:`~matplotlib.axes.Axes`
        Specified starting axes, optional.
    projection : :class:`str`
        Whether to create a projected (e.g. ternary) axes.

    Returns
    --------
    ax : :class:`~matplotlib.axes.Axes`
    """
    if projection is not None:  # e.g. ternary
        if ax is None:
            fig, ax = plt.subplots(
                1,
                subplot_kw=dict(projection=projection),
                **subkwargs(kwargs, plt.subplots, plt.figure)
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
                except ValueError: #ax is not in list
                    # ASSUMPTION due to mis-referencing:
                    # take the first ternary one
                    ax = [a for a in current_axes if a.name =='ternary'][0]
            else:
                pass
    else:
        if ax is None:
            fig, ax = plt.subplots(1, **subkwargs(kwargs, plt.subplots, plt.figure))
    return ax


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
    for ax in axes:
        if which == "x":
            ax.get_shared_x_axes().join(*axes)
        elif which == "y":
            ax.get_shared_y_axes().join(*axes)
        else:
            ax.get_shared_x_axes().join(*axes)
            ax.get_shared_y_axes().join(*axes)


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


def interpolate_path(
    path, resolution=100, periodic=False, aspath=True, closefirst=False, **kwargs
):
    """
    Obtain the interpolation of an existing path at a given
    resolution. Keyword arguments are forwarded to
    :func:`scipy.interpolate.splprep`.

    Parameters
    -----------
    path : :class:`matplotlib.path.Path`
        Path to interpolate.
    resolution :class:`int`
        Resolution at which to obtain the new path. The verticies of
        the new path will have shape (`resolution`, 2).
    periodic : :class:`bool`
        Whether to use a periodic spline.
    periodic : :class:`bool`
        Whether to return a :code:`matplotlib.path.Path`, or simply
        a tuple of x-y arrays.
    closefirst : :class:`bool`
        Whether to first close the path by appending the first point again.

    Returns
    --------
    :class:`matplotlib.path.Path` | :class:`tuple`
        Interpolated :class:`~matplotlib.path.Path` object, if
        `aspath` is :code:`True`, else a tuple of x-y arrays.
    """
    x, y = path.vertices.T
    if closefirst:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    # s=0 forces the interpolation to go through every point
    tck, u = scipy.interpolate.splprep([x[:-1], y[:-1]], s=0, per=periodic, **kwargs)
    xi, yi = scipy.interpolate.splev(np.linspace(0.0, 1.0, resolution), tck)
    # could get control points for path and construct codes here
    codes = None
    pth = matplotlib.path.Path(np.vstack([xi, yi]).T, codes=codes)
    if aspath:
        return pth
    else:
        return pth.vertices.T


def interpolated_patch_path(patch, resolution=100, **kwargs):
    """
    Obtain the periodic interpolation of the existing path of a patch at a
    given resolution.

    Parameters
    -----------
    patch : :class:`matplotlib.patches.Patch`
        Patch to obtain the original path from.
    resolution :class:`int`
        Resolution at which to obtain the new path. The verticies of the new path
        will have shape (`resolution`, 2).

    Returns
    --------
    :class:`matplotlib.path.Path`
        Interpolated :class:`~matplotlib.path.Path` object.
    """
    pth = patch.get_path()
    tfm = patch.get_transform()
    pathtfm = tfm.transform_path(pth)
    return interpolate_path(
        pathtfm, resolution=resolution, aspath=True, periodic=True, **kwargs
    )


def get_contour_paths(ax, resolution=100):
    """
    Extract the paths of contours from a contour plot.

    Parameters
    ------------
    ax : :class:`matplotlib.axes.Axes`
        Axes to extract contours from.
    resolution : :class:`int`
        Resolution of interpolated splines to return.

    Returns
    --------
    contourspaths : :class:`list` (:class:`list`)
        List of lists, each represnting one line collection (a single contour). In the
        case where this contour is multimodal, there will be multiple paths for each
        contour.
    contournames : :class:`list`
        List of names for contours, where they have been labelled, and there are no
        other text artists on the figure.
    contourstyles : :class:`list`
        List of styles for contours.

    Notes
    ------

        This method assumes that contours are the only
        :code:`matplotlib.collections.LineCollection` objects within an axes;
        and when this is not the case, additional non-contour objects will be returned.
    """
    linecolls = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.LineCollection)
    ]
    rgba = [[int(c) for c in lc.get_edgecolors().flatten() * 255] for lc in linecolls]
    styles = [{"color": c} for c in rgba]
    names = [None for lc in linecolls]
    if (len(ax.artists) == len(linecolls)) and all(
        [a.get_text() != "" for a in ax.artists]
    ):
        names = [a.get_text() for a in ax.artists]
    return (
        [
            [
                interpolate_path(p, resolution=resolution, periodic=True, aspath=False)
                for p in lc.get_paths()
            ]
            for lc in linecolls
        ],
        names,
        styles,
    )


def path_to_csv(path, xname="x", yname="y", delim=", ", linesep=os.linesep):
    """
    Extract the verticies from a path and write them to csv.

    Parameters
    ------------
    path : :class:`matplotlib.path.Path` | :class:`tuple`
        Path or x-y tuple to use for coordinates.
    xname : :class:`str`
        Name of the x variable.
    yname : :class:`str`
        Name of the y variable.
    delim : :class:`str`
        Delimiter for the csv file.
    linesep : :class:`str`
        Line separator character.

    Returns
    --------
    :class:`str`
        String-representation of csv file, ready to be written to disk.
    """
    if isinstance(path, matplotlib.path.Path):
        x, y = path.vertices.T
    else:  # isinstance(path, (tuple, list))
        x, y = path

    header = [xname, yname]
    datalines = [[x, y] for (x, y) in zip(x, y)]
    content = linesep.join(
        [delim.join(map(str, line)) for line in [header] + datalines]
    )
    return content


def add_colorbar(mappable, **kwargs):
    """
    Adds a colorbar to a given mappable object.

    Source: http://joseph-long.com/writing/colorbars/

    Parameters
    ----------
    mappable
        The Image, ContourSet, etc. to which the colorbar applies.

    Returns
    ----------
    :class:`matplotlib.colorbar.Colorbar`

    Todo
    ------
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


def bin_centres_to_edges(centres):
    """
    Translates point estimates at the centres of bins to equivalent edges,
    for the case of evenly spaced bins.

    Todo
    ------
        * This can be updated to unevenly spaced bins, just need to calculate outer bins.
    """
    sortcentres = np.sort(centres.flatten())
    step = (sortcentres[1] - sortcentres[0]) / 2.0
    return np.append(sortcentres - step, [sortcentres[-1] + step])


def bin_edges_to_centres(edges):
    """
    Translates edges of histogram bins to bin centres.
    """
    if edges.ndim == 1:
        steps = (edges[1:] - edges[:-1]) / 2
        return edges[:-1] + steps
    else:
        steps = (edges[1:, 1:] - edges[:-1, :-1]) / 2
        centres = edges[:-1, :-1] + steps
        return centres


def affine_transform(mtx=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    """
    Construct a function which will perform a 2D affine transform based on
    a 3x3 affine matrix.

    Parameters
    -----------
    mtx : :class:`numpy.ndarray`
    """

    def tfm(data):
        xy = data[:, :2]
        return (mtx @ np.vstack((xy.T[:2], np.ones(xy.T.shape[1]))))[:2]

    return tfm


def ABC_to_xy(ABC, xscale=1.0, yscale=1.0):
    """
    Convert ternary compositional coordiantes to x-y coordinates
    for visualisation within a triangle.

    Parameters
    -----------
    ABC : :class:`numpy.ndarray`
        Ternary array (:code:`samples, 3`).
    xscale : :class:`float`
        Scale for x-axis.
    yscale : :class:`float`
        Scale for y-axis.

    Returns
    --------
    :class:`numpy.ndarray`
        Array of x-y coordinates (:code:`samples, 2`)
    """
    assert ABC.shape[-1] == 3
    # transform from ternary to xy cartesian
    scale = affine_transform(np.array([[xscale, 0, 0], [0, yscale, 0], [0, 0, 1]]))
    shear = affine_transform(np.array([[1, 1 / 2, 0], [0, 1, 0], [0, 0, 1]]))
    xy = scale(shear(close(ABC)).T)
    return xy.T


def xy_to_ABC(xy, xscale=1.0, yscale=1.0):
    """
    Convert x-y coordinates within a triangle to compositional ternary coordinates.

    Parameters
    -----------
    xy : :class:`numpy.ndarray`
        XY array (:code:`samples, 2`).
    xscale : :class:`float`
        Scale for x-axis.
    yscale : :class:`float`
        Scale for y-axis.

    Returns
    --------
    :class:`numpy.ndarray`
        Array of ternary coordinates (:code:`samples, 3`)
    """
    assert xy.shape[-1] == 2
    # transform from xy cartesian to ternary
    scale = affine_transform(
        np.array([[1 / xscale, 0, 0], [0, 1 / yscale, 0], [0, 0, 1]])
    )
    shear = affine_transform(np.array([[1, -1 / 2, 0], [0, 1, 0], [0, 0, 1]]))
    A, B = shear(scale(xy).T)
    C = 1.0 - (A + B)  # + (xscale-1) + (yscale-1)
    return np.vstack([A, B, C]).T


def ternary_grid(
    data, nbins=10, margin=0.001, force_margin=False, yscale=1.0, tfm=lambda x: x
):
    """
    Construct a grid within a ternary space.

    Parameters
    ------------
    data : :class:`numpy.ndarray`
        Data to construct the grid around (:code:`(samples, 3)`).
    nbins : :class:`int`
        Number of bins for grid.
    margin : :class:`float`
        Proportional value for the position of the outer boundary of the grid.
    forge_margin : :class:`bool`
        Whether to enforce the grid margin.
    yscale : :class:`float`
        Y scale for the specific ternary diagram.
    tfm :
        Log transform to use for the grid creation.

    Returns
    --------
    bins : :class:`numpy.ndarray`
        (:code:`(samples, 3)`)
    binedges : :class:`numpy.ndarray`
        Position of bin edges.
    centregrid : :class:`list` of :class:`numpy.ndarray`
        Meshgrid of bin centres.
    edgegrid : :class:`list` of :class:`numpy.ndarray`
        Meshgrid of bin edges.
    """
    data = close(data)

    if not force_margin:
        margin = min([margin, np.nanmin(data[data > 0])])

    # let's construct a bounding triangle
    bounds = np.array(  # three points defining the edges of what will be rendered
        [
            [margin, margin, 1.0 - 2 * margin],
            [margin, 1.0 - 2 * margin, margin],
            [1.0 - 2 * margin, margin, margin],
        ]
    )
    xbounds, ybounds = ABC_to_xy(bounds, yscale=yscale).T
    xbounds = np.hstack((xbounds, [xbounds[0]]))
    ybounds = np.hstack((ybounds, [ybounds[0]]))
    tck, u = scipy.interpolate.splprep([xbounds, ybounds], per=True, s=0, k=1)
    xi, yi = scipy.interpolate.splev(np.linspace(0, 1.0, 10000), tck)

    A, B, C = xy_to_ABC(np.vstack([xi, yi]).T, yscale=yscale).T
    abcbounds = np.vstack([A, B, C])

    abounds = tfm(abcbounds.T)
    ndim = abounds.shape[1]
    # bins for evaluation
    bins = [
        np.linspace(np.nanmin(abounds[:, dim]), np.nanmax(abounds[:, dim]), nbins)
        for dim in range(ndim)
    ]
    binedges = [bin_centres_to_edges(b) for b in bins]
    centregrid = np.meshgrid(*bins)
    edgegrid = np.meshgrid(*binedges)

    assert len(bins) == ndim
    return bins, binedges, centregrid, edgegrid


def conditional_prob_density(
    y,
    x=None,
    logy=False,
    resolution=5,
    ybins=100,
    rescale=True,
    mode="binkde",
    ret_centres=False,
):
    """
    Estimate the conditional probability density of one dependent variable.

    Parameters
    -----------
    y : :class:`numpy.ndarray`
        Dependent variable for which to calculate conditional probability P(y | X=x)
    x : :class:`numpy.ndarray`, :code:`None`
        Optionally-specified independent index.
    logy : :class:`bool`
        Whether to use a logarithmic bin spacing on the y axis.
    resolution : :class:`int`
        Points added per segment via interpolation along the x axis.
    ybins : :class:`int`
        Bins for histograms and grids along the independent axis.
    rescale : :class:`bool`
        Whether to rescale bins to give the same max Z across x.
    mode : :class:`str`
        Mode of computation.

            If mode is :code:`"ckde"`, use
            :func:`statsmodels.nonparametric.KDEMultivariateConditional` to compute a
            conditional kernel density estimate. If mode is :code:`"kde"`, use a normal
            gaussian kernel density estimate. If mode is :code:`"binkde"`, use a gaussian
            kernel density estimate over y for each bin. If mode is :code:`"hist"`,
            compute a histogram.
    ret_centres : :class:`bool`
        Whether to return bin centres in addtion to histogram edges,
        e.g. for later contouring.

    Returns
    -------
    :class:`tuple` of :class:`numpy.ndarray`
        :code:`x` bin edges :code:`xe`, :code:`y` bin edges :code:`ye`, histogram/density
        estimates :code:`Z`. If :code:`ret_centres` is :code:`True`, the last two return
        values will contain the bin centres :code:`xi`, :code:`yi`.


    Notes
    ------

        * Bins along the x axis are defined such that the x points (including
          interpolated points) are the centres.

    Todo
    -----

        * Tests
        * Implement log grids (for y)
        * Add approach for interpolation? (need resolution etc) - this will resolve lines, not points!
    """
    # check for shapes
    assert not ((x is None) and (y is None))
    if y is None:  # Swap the variables. Create an index for x
        y = x
        x = None

    nvar = y.shape[1]
    if x is None:  # Create a simple arange-based index
        x = np.arange(nvar)

    if not x.shape == y.shape:
        try:  # x is an index to be tiled
            assert y.shape[1] == x.shape[0]
            x = np.tile(x, y.shape[0]).reshape(*y.shape)
        except AssertionError:
            # shape mismatch
            msg = "Mismatched shapes: x: {}, y: {}. Needs either ".format(
                x.shape, y.shape
            )
            raise AssertionError(msg)

    if resolution:
        xy = np.array([x, y])
        xy = np.swapaxes(xy, 1, 0)
        xy = interpolate_line(xy, n=resolution, logy=logy)
        x, y = np.swapaxes(xy, 0, 1)

    xx = np.sort(x[0])
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    # remove non finite values for kde functions
    ystep = [(ymax - ymin) / ybins, (ymax / ymin) / ybins][logy]
    yy = [linspc_, logspc_][logy](ymin, ymax, step=ystep, bins=ybins)
    if logy:  # make grid equally spaced, evaluate in log then transform back
        y, yy = np.log(y), np.log(yy)
    # yy is backwards?
    xi, yi = np.meshgrid(xx, yy)
    xe, ye = np.meshgrid(bin_centres_to_edges(xx), bin_centres_to_edges(yy))

    if mode == "ckde":
        fltr = np.isfinite(y.flatten()) & np.isfinite(x.flatten())
        x, y = x.flatten()[fltr], y.flatten()[fltr]
        if HAVE_SM:
            dens_c = sm.nonparametric.KDEMultivariateConditional(
                endog=[y], exog=[x], dep_type="c", indep_type="c", bw="normal_reference"
            )
        else:
            raise ImportError("Requires statsmodels.")
        # statsmodels pdf takes values in reverse order
        zi = dens_c.pdf(yi.flatten(), xi.flatten()).reshape(xi.shape)
    elif mode == "kde":  # kde of dataset
        xkde = gaussian_kde(x[0])(x[0])  # marginal density along x
        fltr = np.isfinite(y.flatten()) & np.isfinite(x.flatten())
        x, y = x.flatten()[fltr], y.flatten()[fltr]
        try:
            kde = gaussian_kde(np.vstack([x, y]))
        except LinAlgError:  # singular matrix, need to add miniscule noise on x?
            logger.warn("Singular Matrix")
            logger.x = x + np.random.randn(*x.shape) * np.finfo(np.float).eps
            kde = gaussian_kde(np.vstack(([x, y])).T)

        zi = kde(flattengrid([xi, yi]).T).reshape(xi.shape) / xkde[np.newaxis, :]
    elif mode == "binkde":  # calclate a kde per bin
        zi = np.zeros(xi.shape)
        for bin in range(x.shape[1]):
            # if np.isfinite(y[:, bin]).any(): # bins can be empty
            kde = gaussian_kde(y[np.isfinite(y[:, bin]), bin])
            zi[:, bin] = kde(yi[:, bin])
            # else:
            # pass
    elif "hist" in mode.lower():  # simply compute the histogram
        # histogram monotonically increasing bins, requires logbins be transformed
        # calculate histogram in logy if needed

        bins = [bin_centres_to_edges(xx), bin_centres_to_edges(yy)]
        H, xe, ye = np.histogram2d(x.flatten(), y.flatten(), bins=bins)

        zi = H.T.reshape(xi.shape)
    else:
        raise NotImplementedError

    if rescale:  # rescale bins across x
        xzfactors = np.nanmax(zi) / np.nanmax(zi, axis=0)
        zi *= xzfactors[np.newaxis, :]

    if logy:
        yi, ye = np.exp(yi), np.exp(ye)
    if ret_centres:
        return xe, ye, zi, xi, yi
    return xe, ye, zi


def ternary_patch(scale=100.0, yscale=1.0, xscale=1.0, **kwargs):
    """
    Create the background triangle patch for a ternary plot.
    """
    return matplotlib.patches.Polygon(
        ABC_to_xy(np.eye(3), yscale=yscale, xscale=xscale) * scale, **kwargs
    )


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


def rect_from_centre(x, y, dx=0, dy=0, **kwargs):
    """
    Takes an xy point, and creates a rectangular patch centred about it.
    """
    # If either x or y is nan
    if any([np.isnan(i) for i in [x, y]]):
        return None
    if np.isnan(dx):
        dx = 0
    if np.isnan(dy):
        dy = 0
    llc = (x - dx, y - dy)
    return matplotlib.patches.Rectangle(llc, 2 * dx, 2 * dy, **kwargs)


def draw_vector(v0, v1, ax=None, **kwargs):
    """
    Plots an arrow represnting the direction and magnitue of a principal
    component on a biaxial plot.

    Modified after Jake VanderPlas' Python Data Science Handbook
    https://jakevdp.github.io/PythonDataScienceHandbook/ \
    05.09-principal-component-analysis.html

    Todo
    -----
        Update for ternary plots.

    """
    ax = ax
    arrowprops = dict(arrowstyle="->", linewidth=2, shrinkA=0, shrinkB=0)
    arrowprops.update(kwargs)
    ax.annotate("", v1, v0, arrowprops=arrowprops)


def vector_to_line(
    mu: np.array, vector: np.array, variance: float, spans: int = 4, expand: int = 10
):
    """
    Creates an array of points representing a line along a vector - typically
    for principal component analysis. Modified after Jake VanderPlas' Python Data
    Science Handbook https://jakevdp.github.io/PythonDataScienceHandbook/ \
    05.09-principal-component-analysis.html
    """
    length = np.sqrt(variance)
    parts = np.linspace(-spans, spans, expand * spans + 1)
    line = length * np.dot(parts[:, np.newaxis], vector[np.newaxis, :]) + mu
    line = length * parts.reshape(parts.shape[0], 1) * vector + mu
    return line


def plot_stdev_ellipses(
    comp, nstds=4, scale=100, resolution=1000, transform=None, ax=None, **kwargs
):
    """
    Plot covariance ellipses at a number of standard deviations from the mean.

    Parameters
    -------------
    comp : :class:`numpy.ndarray`
        Composition to use.
    nstds : :class:`int`
        Number of standard deviations from the mean for which to plot the ellipses.
    scale : :class:`float`
        Scale applying to all x-y data points. For intergration with python-ternary.
    transform : :class:`callable`
        Function for transformation of data prior to plotting (to either 2D or 3D).
    ax : :class:`matplotlib.axes.Axes`
        Axes to plot on.

    Returns
    -------
    ax :  :class:`matplotlib.axes.Axes`
    """
    mean, cov = np.nanmean(comp, axis=0), nancov(comp)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[::-1]))

    if ax is None:
        projection = None
        if callable(transform) and (transform is not None):
            if transform(comp).shape[1] == 3:
                projection = "ternary"

        fig, ax = plt.subplots(1, subplot_kw=dict(projection=projection))

    for nstd in np.arange(1, nstds + 1)[::-1]:  # backwards for svg construction
        # here we use the absolute eigenvalues
        xsig, ysig = nstd * np.sqrt(np.abs(vals))  # n sigmas
        ell = matplotlib.patches.Ellipse(
            xy=mean.flatten(), width=2 * xsig, height=2 * ysig, angle=theta[:1]
        )
        points = interpolated_patch_path(ell, resolution=resolution).vertices

        if callable(transform) and (transform is not None):
            points = transform(points)  # transform to compositional data

        if points.shape[1] == 3:
            ax_transfrom = (ax.transData + ax.transTernaryAxes.inverted()).inverted()
            points = ax_transfrom.transform(points)  # transform to axes coords

        patch = matplotlib.patches.PathPatch(matplotlib.path.Path(points), **kwargs)
        patch.set_edgecolor("k")
        patch.set_alpha(1.0 / nstd)
        patch.set_linewidth(0.5)
        ax.add_artist(patch)
    return ax


def plot_pca_vectors(comp, nstds=2, scale=100.0, transform=None, ax=None, **kwargs):
    """
    Plot vectors corresponding to principal components and their magnitudes.

    Parameters
    -------------
    comp : :class:`numpy.ndarray`
        Composition to use.
    nstds : :class:`int`
        Multiplier for magnitude of individual principal component vectors.
    scale : :class:`float`
        Scale applying to all x-y data points. For intergration with python-ternary.
    transform : :class:`callable`
        Function for transformation of data prior to plotting (to either 2D or 3D).
    ax : :class:`matplotlib.axes.Axes`
        Axes to plot on.

    Returns
    -------
    ax :  :class:`matplotlib.axes.Axes`

    Todo
    -----
        * Minor reimplementation of the sklearn PCA to avoid dependency.

            https://en.wikipedia.org/wiki/Principal_component_analysis
    """
    pca = PCA(n_components=2)
    pca.fit(comp)

    if ax is None:
        fig, ax = plt.subplots(1)

    for variance, vector in zip(pca.explained_variance_, pca.components_):
        line = vector_to_line(pca.mean_, vector, variance, spans=nstds)
        if callable(transform) and (transform is not None):
            line = transform(line)
        line *= scale
        ax.plot(*line.T, **kwargs)
    return ax


def plot_2dhull(data, ax=None, splines=False, s=0, **plotkwargs):
    """
    Plots a 2D convex hull around an array of xy data points.
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    chull = scipy.spatial.ConvexHull(data, incremental=True)
    x, y = data[chull.vertices].T
    if not splines:
        lines = ax.plot(np.append(x, [x[0]]), np.append(y, [y[0]]), **plotkwargs)
    else:
        # https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
        tck, u = scipy.interpolate.splprep([x, y], per=True, s=s)
        xi, yi = scipy.interpolate.splev(np.linspace(0, 1, 1000), tck)
        lines = ax.plot(xi, yi, **plotkwargs)
    return lines


def get_axis_density_methods(ax):
    """
    Get the relevant density and contouring methods for a given axis.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes` | :class:`mpltern.ternary.TernaryAxes`
        Axis to check.

    Returns
    --------
    pcolor, contour, contourf
        Relevant functions for this axis.
    """
    if ax.name == "ternary":
        pcolor = ax.tripcolor
        contour = ax.tricontour
        contourf = ax.tricontourf
    else:
        if USE_PCOLOR:
            pcolor = ax.pcolor
        else:
            pcolor = ax.pcolormesh
        contour = ax.contour
        contourf = ax.contourf
    return pcolor, contour, contourf


def percentile_contour_values_from_meshz(
    z, percentiles=[0.95, 0.66, 0.33], resolution=1000
):
    """
    Integrate a probability density distribution Z(X,Y) to obtain contours in Z which
    correspond to specified percentile contours.T

    Parameters
    ----------
    z : :class:`numpy.ndarray`
        Probability density function over x, y.
    percentiles : :class:`numpy.ndarray`
        Percentile values for which to create contours.
    resolution : :class:`int`
        Number of bins for thresholds between 0. and max(Z)

    Returns
    -------
    labels : :class:`list`
        Labels for contours (percentiles, if above minimum z value).

    contours : :class:`list`
        Contour height values.
    """
    percentiles = sorted(percentiles, reverse=True)
    # Integral approach from https://stackoverflow.com/a/37932566
    t = np.linspace(0.0, z.max(), resolution)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
    f = scipy.interpolate.interp1d(integral, t)
    try:
        t_contours = f(np.array(percentiles) * z.sum())
        return percentiles, t_contours
    except ValueError:
        logger.debug(
            "Percentile contour below minimum for given resolution"
            "Returning Minimium."
        )
        non_one = integral[~np.isclose(integral, np.ones_like(integral))]
        return ["min"], f(np.array([np.nanmax(non_one)]))


def plot_Z_percentiles(
    *coords,
    zi=None,
    percentiles=[0.95, 0.66, 0.33],
    ax=None,
    extent=None,
    fontsize=8,
    cmap=None,
    contour_labels=None,
    label_contours=True,
    **kwargs
):
    """
    Plot percentile contours onto a 2D  (scaled or unscaled) probability density
    distribution Z over X,Y.

    Parameters
    ------------
    coords : :class:`numpy.ndarray`
        Arrays of (x, y) or (a, b, c) coordinates.
    z : :class:`numpy.ndarray`
        Probability density function over x, y.
    percentiles : :class:`list`
        Percentile values for which to create contours.
    ax : :class:`matplotlib.axes.Axes`, :code:`None`
        Axes on which to plot. If none given, will create a new Axes instance.
    extent : :class:`list`, :code:`None`
        List or np.ndarray in the form [-x, +x, -y, +y] over which the image extends.
    fontsize : :class:`float`
        Fontsize for the contour labels.
    cmap : :class:`matplotlib.colors.ListedColormap`
        Color map for the contours, contour labels and imshow.
    contour_labels : :class:`dict`
        Labels to assign to contours, organised by level.
    label_contours :class:`bool`
        Whether to add text labels to individual contours.

    Returns
    -------
    :class:`matplotlib.contour.QuadContourSet`
        Plotted and formatted contour set.
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 6))

    if extent is None:
        # if len(coords) == 2:  # currently won't work for ternary
        extent = np.array([[np.min(c), np.max(c)] for c in coords[:2]]).flatten()

    clabels, contours = percentile_contour_values_from_meshz(
        zi, percentiles=percentiles
    )

    pcolor, contour, contourf = get_axis_density_methods(ax)
    cs = contour(*coords, zi, levels=contours, cmap=cmap, **kwargs)
    if label_contours:
        fs = kwargs.pop("fontsize", None) or 8
        lbls = ax.clabel(cs, fontsize=fs, inline_spacing=0)
        z_contours = sorted(list(set([float(l.get_text()) for l in lbls])))
        trans = {
            float(t): str(p)
            for t, p in zip(z_contours, sorted(percentiles, reverse=True))
        }
        if contour_labels is None:
            _labels = [trans[float(l.get_text())] for l in lbls]
        else:  # get the labels from the dictionary provided
            contour_labels = {str(k): str(v) for k, v in contour_labels.items()}
            _labels = [contour_labels[trans[float(l.get_text())]] for l in lbls]

        [l.set_text(t) for l, t in zip(lbls, _labels)]
    return cs


def plot_cooccurence(arr, ax=None, normalize=True, log=False, colorbar=False, **kwargs):
    """
    Plot the co-occurence frequency matrix for a given input.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`, :code:`None`
        The subplot to draw on.
    normalize : :class:`bool`
        Whether to normalize the cooccurence to compare disparate variables.
    log : :class:`bool`
        Whether to take the log of the cooccurence.
    colorbar : :class:`bool`
        Whether to append a colorbar.

    Returns
    --------
    :class:`matplotlib.axes.Axes`
        Axes on which the cooccurence plot is added.
    """
    arr = np.array(arr)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(4 + [0.0, 0.2][colorbar], 4))
    co_occur = cooccurence_pattern(arr, normalize=normalize, log=log)
    heatmap = ax.pcolor(co_occur, **kwargs)
    ax.set_yticks(np.arange(co_occur.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(co_occur.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    if colorbar:
        add_colorbar(heatmap, **kwargs)
    return ax


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


def nan_scatter(xdata, ydata, ax=None, axes_width=0.2, **kwargs):
    """
    Scatter plot with additional marginal axes to plot data for which data is partially
    missing. Additional keyword arguments are passed to matplotlib.

    Parameters
    -----------
    xdata : :class:`numpy.ndarray`
        X data
    ydata: class:`numpy.ndarray` | pd.Series
        Y data
    ax : :class:`matplotlib.axes.Axes`
        Axes on which to plot.
    axes_width : :class:`float`
        Width of the marginal axes.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the nan_scatter is plotted.
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.scatter(xdata, ydata, **kwargs)

    if hasattr(ax, "divider"):  # Don't rebuild axes
        div = ax.divider
        nanaxx = div.nanaxx
        nanaxy = div.nanaxy
    else:  # Build axes
        nanaxx = subaxes(ax, side="bottom", width=axes_width)
        nanaxx.invert_yaxis()
        nanaxy = subaxes(ax, side="left", width=axes_width)
        nanaxy.invert_xaxis()

    nanxdata = xdata[(np.isnan(ydata) & np.isfinite(xdata))]
    nanydata = ydata[(np.isnan(xdata) & np.isfinite(ydata))]

    yminmax = np.nanmin(ydata), np.nanmax(ydata)
    no_ybins = 50
    ybinwidth = (np.nanmax(ydata) - np.nanmin(ydata)) / no_ybins
    ybins = np.linspace(np.nanmin(ydata), np.nanmax(ydata) + ybinwidth, no_ybins)

    nanaxy.hist(nanydata, bins=ybins, orientation="horizontal", **kwargs)
    nanaxy.scatter(
        10 * np.ones_like(nanydata) + 5 * np.random.randn(len(nanydata)),
        nanydata,
        zorder=-1,
        **kwargs
    )

    xminmax = np.nanmin(xdata), np.nanmax(xdata)
    no_xbins = 50
    xbinwidth = (np.nanmax(xdata) - np.nanmin(xdata)) / no_xbins
    xbins = np.linspace(np.nanmin(xdata), np.nanmax(xdata) + xbinwidth, no_xbins)

    nanaxx.hist(nanxdata, bins=xbins, **kwargs)
    nanaxx.scatter(
        nanxdata,
        10 * np.ones_like(nanxdata) + 5 * np.random.randn(len(nanxdata)),
        zorder=-1,
        **kwargs
    )

    return ax


def save_figure(
    figure, save_at="", name="fig", save_fmts=["png"], output=False, **kwargs
):
    """
    Save a figure at a specified location in a number of formats.
    """
    default_config = dict(dpi=600, bbox_inches="tight", transparent=True)
    config = default_config.copy()
    config.update(kwargs)
    for fmt in save_fmts:
        out_filename = os.path.join(str(save_at), name + "." + fmt)
        if output:
            logger.info("Saving " + out_filename)
        figure.savefig(out_filename, format=fmt, **config)


def save_axes(ax, save_at="", name="fig", save_fmts=["png"], pad=0.0, **kwargs):
    """
    Save either a single or multiple axes (from a single figure) based on their
    extent. Uses the save_figure procedure to save at a specific location using
    a number of formats.

    Todo
    -----

        * Add legend to items
    """
    # Check if axes is a single axis or list of axes

    if isinstance(ax, matplotlib.axes.Axes):
        extent = get_full_extent(ax, pad=pad)
        figure = ax.figure
    else:
        extent_items = []
        for a in ax:
            extent_items.append(get_full_extent(a, pad=pad))
        figure = ax[0].figure
        extent = Bbox.union([item for item in extent_items])
    save_figure(
        figure,
        bbox_inches=extent,
        save_at=save_at,
        name=name,
        save_fmts=save_fmts,
        **kwargs
    )


def get_full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles. Text objects are first drawn to define the extents.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Axes of which to check items to get full extent.
    pad : :class:`float` | :class:`tuple`
        Amount of padding to add to the full extent prior to returning. If a tuple is
        passed, the padding will be as above, but for x and y directions, respectively.

    Returns
    --------
    :class:`matplotlib.transforms.Bbox`
        Bbox of the axes with optional additional padding.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.renderer

    items = [ax]

    if len(ax.get_title()):
        items += [ax.title]

    for a in [ax.xaxis, ax.yaxis]:
        if len(a.get_label_text()):
            items += [a.label]

    for t_lb in [ax.get_xticklabels(), ax.get_yticklabels()]:
        if np.array([len(i.get_text()) > 0 for i in t_lb]).any():
            items += t_lb

    bbox = Bbox.union([item.get_window_extent(renderer) for item in items])
    if isinstance(pad, (float, int)):
        full_extent = bbox.expanded(1.0 + pad, 1.0 + pad)
    elif isinstance(pad, (list, tuple)):
        full_extent = bbox.expanded(1.0 + pad[0], 1.0 + pad[1])
    else:
        raise NotImplementedError
    return full_extent.transformed(ax.figure.dpi_scale_trans.inverted())
