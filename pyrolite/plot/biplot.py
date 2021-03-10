import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..comp import codata
from ..util.plot.axes import init_axes
from ..util.log import Handle

logger = Handle(__name__)


def compositional_SVD(X: np.ndarray):
    """
    Breakdown a set of compositions to vertexes and cases for adding to a
    compositional biplot.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Compositional array.

    Returns
    ---------
    vertexes, cases : :class:`numpy.ndarray`, :class:`numpy.ndarray`
    """
    U, K, V = np.linalg.svd(codata.CLR(X))
    N = X.shape[1]  # dimensionality
    vertexes = K * V.T / (N - 1) ** 0.5
    cases = (N - 1) ** 0.5 * U.T
    return vertexes, cases


def plot_origin_to_points(
    xs,
    ys,
    labels=None,
    ax=None,
    origin=(0, 0),
    color="k",
    marker="o",
    pad=0.05,
    **kwargs
):
    """
    Plot lines radiating from a specific origin. Fornulated for creation of
    biplots (:func:`covariance_biplot`, :func:`compositional_biplot`).

    Parameters
    -----------
    xs, ys : :class:`numpy.ndarray`
        Coordinates for points to add.
    labels : :class:`list`
        Labels for verticies.
    ax : :class:`matplotlib.axes.Axes`
        Axes to plot on.
    origin : :class:`tuple`
        Origin to plot from.
    color : :class:`str`
        Line color to use.
    marker : :class:`str`
        Marker to use for ends of vectors and origin.
    pad : :class:`float`
        Fraction of vector to pad text label.

    Returns
    --------
    :class:`matplotlib.axes.Axes`
        Axes on which radial plot is added.
    """
    x0, y0 = origin
    ax = init_axes(ax=ax, **kwargs)
    _xs, _ys = (
        np.vstack([x0 * np.ones_like(xs), xs]),
        np.vstack([y0 * np.ones_like(ys), ys]),
    )
    ax.plot(_xs, _ys, color=color, marker=marker, **kwargs)

    if labels is not None:
        for ix, label in enumerate(labels):
            x, y = xs[ix], ys[ix]
            dx, dy = x - x0, y - y0
            theta = np.rad2deg(np.arctan(dy / dx))

            x += pad * dx
            y += pad * dy

            if np.abs(theta) > 60:
                ha = "center"
            else:
                ha = ["right" if x < x0 else "left"][0]

            if np.abs(theta) < 30:
                va = "center"
            else:
                va = ["top" if y < y0 else "bottom"][0]
            ax.annotate(label, (x, y), ha=ha, va=va, rotation=theta)

    return ax


def compositional_biplot(data, labels=None, ax=None, **kwargs):
    """
    Create a compositional biplot.

    Parameters
    -----------
    data : :class:`numpy.ndarray`
        Coordinates for points to add.
    labels : :class:`list`
        Labels for verticies.
    ax : :class:`matplotlib.axes.Axes`
        Axes to plot on.

    Returns
    --------
    :class:`matplotlib.axes.Axes`
        Axes on which biplot is added.
    """

    ax = init_axes(ax=ax, **kwargs)

    v, c = compositional_SVD(data)
    ax.scatter(*c[:, :2].T, **kwargs)
    plot_origin_to_points(
        *v[:, :2].T,
        ax=ax,
        marker=None,
        labels=labels,
        alpha=0.5,
        zorder=-1,
        label="Variables",
    )
    return ax
