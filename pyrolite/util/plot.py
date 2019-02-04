import os
from copy import copy
from types import MethodType
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats.kde import gaussian_kde
from scipy.spatial import ConvexHull
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as matax
from matplotlib.transforms import Bbox
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()


def add_legend_items(ax):

    handles_original = []
    handles_original += ax.lines + ax.patches + ax.collections + ax.containers
    # support parasite axes:
    if hasattr(ax, "parasites"):
        for axx in ax.parasites:
            handles_original += (
                axx.lines + axx.patches + axx.collections + axx.containers
            )
    handles, labels = [], []
    for handle in handles_original:
        label = handle.get_label()
        if label and not label.startswith("_"):
            handles.append(handle)
            labels.append(label)


def modify_legend_handles(ax, **kwargs):
    """
    Modify the handles of a legend based for a single axis.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis for which to obtain modifed legend handles.
    kwargs:
        Keyword arguments to be passed to the handles.

    Returns
    -------
    tuple
        Handles, labels to be passed to a legend call.
    """
    hndls, labls = ax.get_legend_handles_labels()
    _hndls = []
    for h in hndls:
        _h = copy(h)
        _h.update(kwargs)
        _hndls.append(_h)
    return _hndls, labls


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
    matplotlib.colorbar.Colorbar
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax, **kwargs)


def ABC_to_tern_xy(ABC):
    (A, B, C) = ABC
    T = A + B + C
    A_n, B_n, C_n = np.divide(A, T), np.divide(B, T), np.divide(C, T)
    xdata = 100.0 * (
        (C_n / np.sin(np.pi / 3) + A_n / np.tan(np.pi / 3.0)) * np.sin(np.pi / 3.0)
    )
    ydata = 100.0 * (2.0 / (3.0 ** 0.5)) * A_n * np.sin(np.pi / 3.0)
    return xdata, ydata


def tern_heatmapcoords(data, scale=10, bins=10):
    # this appears to cause problems for ternary density diagrams
    x, y = ABC_to_tern_xy(data)
    xydata = np.vstack((x, y))
    k = gaussian_kde(xydata)

    tridata = dict()
    step = scale // bins
    for i in np.arange(0, scale + 1, step):
        for j in np.arange(0, scale + 1 - i, step):
            datacoord = i, j
            # datacoord = i+0.5*step, j+0.5*step
            tridata[(i, j)] = np.float(k(np.vstack(datacoord)))

    return tridata


def proxy_rect(**kwargs):
    """
    Generates a legend proxy for a filled region.

    Returns
    ----------
    matplotlib.patches.Rectangle
    """
    return patches.Rectangle((0, 0), 1, 1, **kwargs)


def proxy_line(**kwargs):
    """
    Generates a legend proxy for a line region.

    Returns
    ----------
    matplotlib.lines.Line2D
    """
    return mlines.Line2D(range(1), range(1), **kwargs)


def draw_vector(v0, v1, ax=None, **kwargs):
    """
    Plots an arrow represnting the direction and magnitue of a principal
    component on a biaxial plot.

    Todo: update for ternary plots.

    Modified after Jake VanderPlas' Python Data Science Handbook
    https://jakevdp.github.io/PythonDataScienceHandbook/ \
    05.09-principal-component-analysis.html
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
    parts = np.linspace(-spans, spans, expand * 2 * spans + 1)
    line = length * np.dot(parts[:, np.newaxis], vector[np.newaxis, :]) + mu
    line = length * parts.reshape(parts.shape[0], 1) * vector + mu
    return line


def plot_2dhull(ax, data, splines=False, s=0, **plotkwargs):
    """
    Plots a 2D convex hull around an array of xy data points.
    """
    chull = ConvexHull(data, incremental=True)
    x, y = data[chull.vertices].T
    if not splines:
        lines = ax.plot(np.append(x, [x[0]]), np.append(y, [y[0]]), **plotkwargs)
    else:
        # https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
        tck, u = interpolate.splprep([x, y], per=True, s=s)
        xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
        lines = ax.plot(xi, yi, **plotkwargs)
    return lines


def percentile_contour_values_from_meshz(
    z, percentiles=[0.95, 0.66, 0.33], resolution=1000
):
    """
    Integrate a probability density distribution Z(X,Y) to obtain contours in Z which
    correspond to specified percentile contours.T

    Parameters
    ----------
    z : np.ndarray
        Probability density function over x, y.
    percentiles : list-like
        Percentile values for which to create contours.
    resolution : int
        Number of bins for thresholds between 0. and max(Z)

    Returns
    -------
    labels
        Labels for contours (percentiles, if above minimum z value).

    contours
        Contour heigh values.
    """
    percentiles = sorted(percentiles, reverse=True)
    # Integral approach from https://stackoverflow.com/a/37932566
    t = np.linspace(0.0, z.max(), resolution)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
    f = interpolate.interp1d(integral, t)
    try:
        t_contours = f(np.array(percentiles) * z.sum())
        return percentiles, t_contours
    except ValueError:
        logger.debug('Percentile contour below minimum for given resolution' \
                     'Returning Minimium.')
        non_one = integral[~np.isclose(integral, np.ones_like(integral))]
        return ['min'], f(np.array([np.nanmax(non_one)]))


def plot_Z_percentiles(
    xi,
    yi,
    zi,
    percentiles=[0.95, 0.66, 0.33],
    ax=None,
    extent=None,
    label_contours=True,
    **kwargs
):
    """
    Plot percentile contours onto a 2D  (scaled or unscaled) probability density
    distribution Z over X,Y.

    Parameters
    ------------
    z : np.ndarray
        Probability density function over x, y.
    percentiles : list-like
        Percentile values for which to create contours.
    ax : matplotlib.axes.Axes
        Axes on which to plot. If none given, will create a new Axes instance.
    extent
        List or np.ndarray in the form [-x, +x, -y, +y] over which the image extends.
    fontsize : np.number
        Fontsize for the contour labels.
    cmap
        Color map for the contours, contour labels and imshow.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        Plotted and formatted contour set.
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 6))

    if extent is None:
        xmin, xmax = np.min(xi), np.max(xi)
        ymin, ymax = np.min(yi), np.max(yi)
        extent = [xmin, xmax, ymin, ymax]

    clabels, contours = percentile_contour_values_from_meshz(zi, percentiles=percentiles)
    cs = ax.contour(xi, yi, zi, levels=contours, extent=extent,  **kwargs)
    if label_contours:
        fs = kwargs.pop('fontsize', None) or 8
        lbls = ax.clabel(cs, fontsize=fs)
        z_contours = sorted(list(set([float(l.get_text()) for l in lbls])))
        trans = {
            float(t): str(p) for t, p in zip(z_contours, sorted(percentiles, reverse=True))
        }
        [l.set_text(trans[float(l.get_text())]) for ix, l in enumerate(lbls)]
    return cs


def nan_scatter(xdata, ydata, ax=None, axes_width=0.2, **kwargs):
    """
    Scatter plot with additional marginal axes to plot data for which data is partially
    missing. Additional keyword arguments are passed to matplotlib.

    Parameters
    -----------
    xdata : np.ndarray | pd.Series
        X data
    ydata: np.ndarray | pd.Series
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
        fig, ax = plt.subplots(1)

    ax.scatter(xdata, ydata, **kwargs)

    if hasattr(ax, "divider"):  # Don't rebuild axes
        div = ax.divider
        nanaxx = div.nanaxx
        nanaxy = div.nanaxy
    else:  # Build axes
        ax.yaxis.set_tick_params(labelleft=False, left=False)
        ax.xaxis.set_tick_params(labelbottom=False, bottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        div = make_axes_locatable(ax)
        ax.divider = div

        nanaxx = div.append_axes("bottom", axes_width, pad=0, sharex=ax)
        div.nanaxx = nanaxx
        nanaxx.invert_yaxis()
        nanaxx.yaxis.set_visible(False)
        nanaxx.spines["left"].set_visible(False)
        nanaxx.spines["right"].set_visible(False)
        nanaxx.set_facecolor("none")

        nanaxy = div.append_axes("left", axes_width, pad=0, sharey=ax)
        div.nanaxy = nanaxy
        nanaxy.invert_xaxis()
        nanaxy.xaxis.set_visible(False)
        nanaxy.spines["top"].set_visible(False)
        nanaxy.spines["bottom"].set_visible(False)
        nanaxy.set_facecolor("none")

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
        out_filename = os.path.join(save_at, name + "." + fmt)
        if output:
            print("Saving " + out_filename)
        figure.savefig(out_filename, format=fmt, **config)


def save_axes(ax, save_at="", name="fig", save_fmts=["png"], pad=0.0, **kwargs):
    """
    Save either a single or multiple axes (from a single figure) based on their
    extent. Uses the save_figure procedure to save at a specific location using
    a number of formats.
    """
    # Check if axes is a single axis or list of axes

    if isinstance(ax, matax.Axes):
        extent = get_full_extent(ax, pad=pad)
        figure = ax.figure
    else:
        extent_items = []
        for a in ax:
            extent_items.append(get_full_extent(a, pad=pad))
        figure = axes[0].figure
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
    ax : matplotlib.axes.Axes
        Axes of which to check items to get full extent.
    pad : np.number
        Amount of padding to add to the full extent prior to returning.

    Returns
    --------
    matplotlib.transforms.Bbox
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
    full_extent = bbox.expanded(1.0 + pad, 1.0 + pad)
    return full_extent.transformed(ax.figure.dpi_scale_trans.inverted())
