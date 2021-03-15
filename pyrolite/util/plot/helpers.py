"""
matplotlib helper functions for commong drawing tasks.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.spatial
from ..math import eigsorted, nancov
from ..text import int_to_alpha
from ..missing import cooccurence_pattern
from .interpolation import interpolated_patch_path
from .axes import add_colorbar, subaxes
from ..log import Handle

logger = Handle(__name__)

try:
    from sklearn.decomposition import PCA
except ImportError:
    msg = "scikit-learn not installed"
    logger.warning(msg)


def alphalabel_subplots(ax, fmt="{}", xy=(0.03, 0.95), ha="left", va="top", **kwargs):
    """
    Add alphabetical labels to a successive series of subplots with a specified format.

    Parameters
    -----------
    ax : :class:`list` | :class:`numpy.ndarray` | :class:`numpy.flatiter`
        Axes to label, in desired order.
    fmt : :class:`str`
        Format string to use. To add e.g. parentheses, you could specify :code:`"({})"`.
    xy : :class:`tuple`
        Position of the labels in axes coordinates.
    ha : :class:`str`
        Horizontal alignment of the labels (:code:`{"left", "right"}`).
    va : :class:`str`
        Vertical alignment of the labels (:code:`{"top", "bottom"}`).
    """
    flat = np.array(ax).flatten()
    # get axes in case of iterator which is consumed
    _ax = [(ix, flat[ix]) for ix in range(len(flat))]
    labels = [(a, fmt.format(int_to_alpha(ix))) for ix, a in _ax]
    [
        a.annotate(label, xy=xy, xycoords=a.transAxes, ha=ha, va=va, **kwargs)
        for a, label in labels
    ]


def get_centroid(poly):
    """
    Centroid of a closed polygon using the Shoelace formula.

    Parameters
    ----------
    poly : :class:`matplotlib.patches.Polygon`
        Polygon to obtain the centroid of.

    Returns
    -------
    cx, cy : :class:`tuple`
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


def nan_scatter(xdata, ydata, ax=None, axes_width=0.2, **kwargs):
    """
    Scatter plot with additional marginal axes to plot data for which data is partially
    missing. Additional keyword arguments are passed to matplotlib.

    Parameters
    ----------
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
        **kwargs
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
        **kwargs
    )

    return ax
