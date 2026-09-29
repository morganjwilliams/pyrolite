import matplotlib.axes
import numpy as np

from ..comp import codata
from ..util.log import Handle
from ..util.plot.axes import init_axes

logger = Handle(__name__)


def compositional_SVD(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Breakdown a set of compositions to vertexes and cases for adding to a
    compositional biplot.

    Parameters
    ----------
    X : numpy.ndarray
        Compositional array.

    Returns
    ---------
    vertexes, cases : numpy.ndarray, numpy.ndarray
    """
    U, K, V = np.linalg.svd(codata.CLR(X))
    N = X.shape[1]  # dimensionality
    vertexes = K * V.T / (N - 1) ** 0.5
    cases = (N - 1) ** 0.5 * U.T
    return vertexes, cases


def plot_origin_to_points(
    xs: np.ndarray,
    ys: np.ndarray,
    labels: list[str] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    color: str = "k",
    marker: str = "o",
    pad: float = 0.05,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot lines radiating from a specific origin. Fornulated for creation of
    biplots (`covariance_biplot`, `compositional_biplot`).

    Parameters
    -----------
    xs, ys : numpy.ndarray
        Coordinates for points to add.
    labels : list
        Labels for verticies.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    origin : tuple
        Origin to plot from.
    color : str
        Line color to use.
    marker : str
        Marker to use for ends of vectors and origin.
    pad : float
        Fraction of vector to pad text label.

    Returns
    --------
    matplotlib.axes.Axes
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


def compositional_biplot(
    data: np.ndarray,
    labels: list[str] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Create a compositional biplot.

    Parameters
    -----------
    data : numpy.ndarray
        Coordinates for points to add.
    labels : list
        Labels for verticies.
    ax : matplotlib.axes.Axes
        Axes to plot on.

    Returns
    --------
    matplotlib.axes.Axes
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
