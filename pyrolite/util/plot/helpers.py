"""
matplotlib helper functions for commong drawing tasks.
"""

from collections.abc import Callable

import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

from pyrolite.util.skl.helpers import HAVE_SKL

from ...comp.codata import inverse_sphere
from ..log import Handle
from ..math import eigsorted, nancov
from ..missing import cooccurence_pattern
from ..text import int_to_alpha
from .axes import add_colorbar, init_axes, subaxes
from .center import visual_center
from .interpolation import interpolated_patch_path

logger = Handle(__name__)

try:
    from sklearn.decomposition import PCA

    HAVE_SKL = True
except ImportError:
    HAVE_SKL = False


def alphalabel_subplots(
    ax: matplotlib.axes.Axes,
    fmt: str = "{}",
    xy: tuple[float, float] = (0.03, 0.95),
    ha: str = "left",
    va: str = "top",
    **kwargs,
):
    """
    Add alphabetical labels to a successive series of subplots with a specified format.

    Parameters
    -----------
    ax : list | numpy.ndarray | numpy.flatiter
        Axes to label, in desired order.
    fmt : str
        Format string to use. To add e.g. parentheses, you could specify `"({})"`.
    xy : tuple
        Position of the labels in axes coordinates.
    ha : str
        Horizontal alignment of the labels (`{"left", "right"}`).
    va : str
        Vertical alignment of the labels (`{"top", "bottom"}`).
    """
    flat = np.array(ax).flatten()
    # get axes in case of iterator which is consumed
    _ax = [(ix, flat[ix]) for ix in range(len(flat))]
    labels = [(a, fmt.format(int_to_alpha(ix))) for ix, a in _ax]
    [
        a.annotate(label, xy=xy, xycoords=a.transAxes, ha=ha, va=va, **kwargs)
        for a, label in labels
    ]


def get_centroid(poly: matplotlib.patches.Polygon) -> tuple[float, float]:
    """
    Centroid of a closed polygon using the Shoelace formula.

    Parameters
    ----------
    poly : matplotlib.patches.Polygon
        Polygon to obtain the centroid of.

    Returns
    -------
    cx, cy : tuple
        Centroid coordinates.
    """
    # get signed area
    verts = poly.get_xy()
    A = 0
    cx, cy = 0, 0
    x, y = verts.T
    for i in range(len(verts) - 1):
        A += x[i] * y[i + 1] - x[i + 1] * y[i]
        cx += (x[i] + x[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
        cy += (y[i] + y[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    A /= 2
    cx /= 6 * A
    cy /= 6 * A
    return cx, cy


def get_visual_center(
    poly: matplotlib.patches.Polygon, vertical_exaggeration: float = 1
) -> tuple[float, float]:
    """
    Visual center of a closed polygon.

    Parameters
    ----------
    poly : matplotlib.patches.Polygon
        Polygon for which to obtain the visual center.

    vertical_exaggeration : float
        Apparent vertical exaggeration of the plot
        (pixels per unit in y direction divided by pixels
        per unit in the x direction).

    Returns
    -------
    cx, cy : tuple
        Centroid coordinates.
    """
    poly_scaled = np.array([poly.get_xy() * [1.0, vertical_exaggeration]])
    x, y = visual_center(poly_scaled)
    return (x, y / vertical_exaggeration)


def rect_from_centre(
    x: float, y: float, dx: float = 0, dy: float = 0, **kwargs
) -> matplotlib.patches.Rectangle:
    """
    Takes an xy point, and creates a rectangular patch centred about it.
    """
    # If either x or y is nan
    if any(np.isnan(i) for i in [x, y]):
        return None
    if np.isnan(dx):
        dx = 0
    if np.isnan(dy):
        dy = 0
    llc = (x - dx, y - dy)
    return matplotlib.patches.Rectangle(llc, 2 * dx, 2 * dy, **kwargs)


def draw_vector(
    v0: np.ndarray, v1: np.ndarray, ax: matplotlib.axes.Axes | None = None, **kwargs
):
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
    arrowprops = {"arrowstyle": "->", "linewidth": 2, "shrinkA": 0, "shrinkB": 0}
    arrowprops.update(kwargs)
    ax.annotate("", v1, v0, arrowprops=arrowprops)


def vector_to_line(
    mu: np.ndarray,
    vector: np.ndarray,
    variance: float,
    spans: int = 4,
    expand: int = 10,
) -> np.ndarray:
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
    comp: np.ndarray,
    nstds: float = 4,
    scale: float = 100,
    resolution: int = 1000,
    transform: Callable | None = None,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot covariance ellipses at a number of standard deviations from the mean.

    Parameters
    -------------
    comp : numpy.ndarray
        Composition to use.
    nstds : int
        Number of standard deviations from the mean for which to plot the ellipses.
    scale : float
        Scale applying to all x-y data points. For intergration with python-ternary.
    transform : Callable
        Function for transformation of data prior to plotting (to either 2D or 3D).
    ax : matplotlib.axes.Axes
        Axes to plot on.

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """
    mean, cov = np.nanmean(comp, axis=0), nancov(comp)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[::-1]))[0]

    if ax is None:
        projection = None
        if callable(transform) and (transform is not None):
            if transform(comp).shape[1] == 3:
                projection = "ternary"

        _fig, ax = plt.subplots(1, subplot_kw={"projection": projection})

    for nstd in np.arange(1, nstds + 1)[::-1]:  # backwards for svg construction
        # here we use the absolute eigenvalues
        xsig, ysig = nstd * np.sqrt(np.abs(vals))  # n sigmas
        ell = matplotlib.patches.Ellipse(
            xy=mean.flatten(),
            width=2 * xsig,
            height=2 * ysig,
            angle=theta,
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


def plot_2dhull(
    data: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    splines: bool = False,
    s: int = 0,
    **plotkwargs,
) -> list[matplotlib.lines.Line2D]:
    """
    Plots a 2D convex hull around an array of xy data points.
    """
    if ax is None:
        _fig, ax = plt.subplots(1)
    chull = scipy.spatial.ConvexHull(data, incremental=True)
    x, y = data[chull.vertices].T
    if not splines:
        lines = ax.plot(np.append(x, [x[0]]), np.append(y, [y[0]]), **plotkwargs)
    else:
        # https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
        tck, _u = scipy.interpolate.splprep([x, y], per=True, s=s)
        xi, yi = scipy.interpolate.splev(np.linspace(0, 1, 1000), tck)
        lines = ax.plot(xi, yi, **plotkwargs)
    return lines


def plot_cooccurence(
    arr: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    normalize: bool = True,
    log: bool = False,
    colorbar: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot the co-occurence frequency matrix for a given input.

    Parameters
    -----------
    ax : matplotlib.axes.Axes, `None`
        The subplot to draw on.
    normalize : bool
        Whether to normalize the cooccurence to compare disparate variables.
    log : bool
        Whether to take the log of the cooccurence.
    colorbar : bool
        Whether to append a colorbar.

    Returns
    --------
    matplotlib.axes.Axes
        Axes on which the cooccurence plot is added.
    """
    arr = np.array(arr)
    if ax is None:
        _fig, ax = plt.subplots(1, figsize=(4 + [0.0, 0.2][colorbar], 4))
    co_occur = cooccurence_pattern(arr, normalize=normalize, log=log)
    heatmap = ax.pcolor(co_occur, **kwargs)
    ax.set_yticks(np.arange(co_occur.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(co_occur.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    if colorbar:
        add_colorbar(heatmap, **kwargs)
    return ax


def nan_scatter(
    xdata: np.ndarray,
    ydata: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    axes_width: float = 0.2,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Scatter plot with additional marginal axes to plot data for which data is partially
    missing. Additional keyword arguments are passed to matplotlib.

    Parameters
    ----------
    xdata : numpy.ndarray
        X data
    ydata: class:numpy.ndarray | pd.Series
        Y data
    ax : matplotlib.axes.Axes
        Axes on which to plot.
    axes_width : float
        Width of the marginal axes.

    Returns
    -------
    matplotlib.axes.Axes
        Axes on which the nan_scatter is plotted.

    """
    if ax is None:
        _fig, ax = plt.subplots(1)

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
        ax.divider.nanaxx = nanaxx  # assign for later use
        ax.divider.nanaxy = nanaxy

    nanxdata = xdata[(np.isnan(ydata) & np.isfinite(xdata))]
    nanydata = ydata[(np.isnan(xdata) & np.isfinite(ydata))]

    # yminmax = np.nanmin(ydata), np.nanmax(ydata)
    no_ybins = 50
    ybinwidth = (np.nanmax(ydata) - np.nanmin(ydata)) / no_ybins
    ybins = np.linspace(np.nanmin(ydata), np.nanmax(ydata) + ybinwidth, no_ybins)

    nanaxy.hist(nanydata, bins=ybins, orientation="horizontal", **kwargs)
    nanaxy.scatter(
        10 * np.ones_like(nanydata) + 5 * np.random.randn(len(nanydata)),
        nanydata,
        zorder=-1,
        **kwargs,
    )

    # xminmax = np.nanmin(xdata), np.nanmax(xdata)
    no_xbins = 50
    xbinwidth = (np.nanmax(xdata) - np.nanmin(xdata)) / no_xbins
    xbins = np.linspace(np.nanmin(xdata), np.nanmax(xdata) + xbinwidth, no_xbins)

    nanaxx.hist(nanxdata, bins=xbins, **kwargs)
    nanaxx.scatter(
        nanxdata,
        10 * np.ones_like(nanxdata) + 5 * np.random.randn(len(nanxdata)),
        zorder=-1,
        **kwargs,
    )

    return ax


###############################################################################
# Helpers for pyrolite.comp.codata.sphere and related functions


def _get_spherical_vector(phis: np.ndarray) -> np.ndarray:
    """
    Get a line aligned to a unit vector corresponding to a specific combination
    of angles.

    Parameters
    ----------
    phis : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    vector = np.sqrt(inverse_sphere(phis))
    return np.vstack([np.zeros_like(vector), vector, vector * 1.5])


def _plot_spherical_vector(
    ax: matplotlib.axes.Axes,
    phis: np.ndarray,
    marker: str = "D",
    markevery: int | tuple[int, int] = (1, 2),
    ls: str = "--",
    **kwargs,
):
    """
    Plot a unit vector corresponding to angles `phis` on a specified axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
    """
    vector = _get_spherical_vector(phis)
    ax.plot(*vector.T, marker=marker, markevery=markevery, ls=ls, **kwargs)


def _get_spherical_arc(thetas0: np.ndarray, thetas1: np.ndarray, resolution: int = 100):
    """
    Get a 3D arc on a sphere between two points.

    Parameters
    ----------
    thetas0 : numpy.ndarray
        Angles corresponding to first unit vector.
    thetas1 : numpy.ndarray
        Angles corresponding to second unit vector.
    resolution : int
        Resolution of the line to be used/number of points in the line.

    Returns
    -------
    numpy.ndarray
    """
    # check that the points are on the sphere?
    v0, v1 = _get_spherical_vector(thetas0)[1], _get_spherical_vector(thetas1)[1]
    vs = v0 + np.linspace(0, 1, resolution + 1)[:, None] * (v1 - v0)
    r = np.sqrt((vs**2).sum(axis=1))  # equivalent arc radius
    vs = vs / r[:, None]
    return vs


def init_spherical_octant(
    angle_indicated: float = 30,
    labels: list[str] = None,
    view_init: tuple[float, float] = (25, 55),
    fontsize: float = 10,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Initalize a figure with a 3D octant of a unit sphere, appropriately labeled
    with regard to angles corresponding to the handling of the respective
    compositional data transformation function (`~pyrolite.comp.codata.sphere`).

    Parameters
    -----------
    angle_indicated : float
        Angle relative to axes for indicating the relative positioning, optional.
    labels : list
        Optional specification of data/axes labels. This will be used for labelling
        of both the axes and optionally-added arcs specifying which angles are
        represented.

    Returns
    -------
    ax : matplotlib.axes.Axes3D
        Initialized 3D axis.
    """
    ax = init_axes(subplot_kw={"projection": "3d"}, **kwargs)

    ax.view_init(*view_init)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.grid(False)

    if labels is None:
        labels = ["x", "y", "z"]
        angle_labels = [r"$\theta_2$", r"$\theta_3$"]
    else:
        angle_labels = [
            r"$\theta_{" + labels[-2] + "}$",
            r"$\theta_{" + labels[-1] + "}$",
        ]

    # axes lines
    lines = np.array([[0, 1, 1.5], [0, 0, 0]])

    ax.plot(*lines[[0, 1, 1]], lw=2, color="k", marker="D", markevery=(1, 2))  # x axis
    ax.plot(*lines[[1, 0, 1]], lw=2, color="k", marker="D", markevery=(1, 2))  # y axis
    ax.plot(*lines[[1, 1, 0]], lw=2, color="k", marker="D", markevery=(1, 2))  # z axis
    # axes labels
    for ix, row in enumerate(np.eye(3) * 1.6):
        ax.text(*row, labels[ix], fontsize=fontsize)

    if angle_indicated is not None:
        _a = np.deg2rad(angle_indicated)
        # theta 2 ##############################################################
        _plot_spherical_vector(ax, np.array([[_a, np.pi / 2]]), color="purple")
        ax.plot(
            *_get_spherical_arc(
                np.array([[_a, np.pi / 2]]), np.array([[0, np.pi / 2]])
            ).T,
            color="purple",
        )
        theta2_pos = (
            _get_spherical_vector(np.array([[_a, np.pi / 2]]))[1]
            + np.array([0, 1, 0]) / 2
        )
        ax.text(*theta2_pos, angle_labels[0], color="purple", fontsize=fontsize)

        # theta 3 ##############################################################
        _plot_spherical_vector(ax, np.array([[_a, _a]]), color="g")
        ax.plot(
            *_get_spherical_arc(np.array([[np.pi / 2, 0]]), np.array([[_a, _a]])).T,
            color="green",
        )
        theta3_pos = (
            _get_spherical_vector(np.array([[_a, _a]]))[1] + np.array([0, 0, 1]) / 2
        )
        ax.text(
            *theta3_pos, angle_labels[1], ha="left", color="green", fontsize=fontsize
        )

    return ax


if HAVE_SKL:

    def plot_pca_vectors(
        comp: np.ndarray,
        nstds: float = 2,
        scale: float = 100.0,
        transform: Callable | None = None,
        ax: matplotlib.axes.Axes | None = None,
        colors: str | list[str] | None = None,
        linestyles: str | list[str] | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Plot vectors corresponding to principal components and their magnitudes.

        Parameters
        -------------
        comp : numpy.ndarray
            Composition to use.
        nstds : int
            Multiplier for magnitude of individual principal component vectors.
        scale : float
            Scale applying to all x-y data points. For intergration with python-ternary.
        transform : Callable
            Function for transformation of data prior to plotting (to either 2D or 3D).
        ax : matplotlib.axes.Axes
            Axes to plot on.

        Returns
        -------
        ax :  matplotlib.axes.Axes

        Todo
        -----
            * Minor reimplementation of the sklearn PCA to avoid dependency.

                https://en.wikipedia.org/wiki/Principal_component_analysis
        """
        pca = PCA(n_components=2)
        pca.fit(comp)

        if ax is None:
            _fig, ax = plt.subplots(1)

        items = [pca.explained_variance_, pca.components_]
        if linestyles is not None:
            assert len(linestyles) == 2
            items.append(linestyles)
        else:
            items.append([None, None])
        if colors is not None:
            assert len(colors) == 2
            items.append(colors)
        else:
            items.append([None, None])
        for variance, vector, linestyle, color in zip(*items):
            line = vector_to_line(pca.mean_, vector, variance, spans=nstds)
            if callable(transform) and (transform is not None):
                line = transform(line)
            line *= scale
            kw = {**kwargs}
            if color is not None:
                kw["color"] = color
            if linestyle is not None:
                kw["ls"] = linestyle
            ax.plot(*line.T, **kw)
        return ax
