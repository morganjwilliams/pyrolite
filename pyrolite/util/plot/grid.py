"""
Gridding and binning functions.
"""

from collections.abc import Callable

import numpy as np
import scipy.interpolate

from ...comp.codata import close
from ..log import Handle
from .transform import ABC_to_xy, xy_to_ABC

logger = Handle(__name__)


def bin_centres_to_edges(centres: np.ndarray, sort: bool = True) -> np.ndarray:
    """
    Translates point estimates at the centres of bins to equivalent edges,
    for the case of evenly spaced bins.

    Todo
    ------
        * This can be updated to unevenly spaced bins, just need to calculate outer bins.
    """
    if sort:
        centres = np.sort(centres.flatten())
    internal_means = (centres[1:] + centres[:-1]) / 2.0
    before, after = (
        centres[0] - (internal_means[0] - centres[0]),
        centres[-1] + (centres[-1] - internal_means[-1]),
    )
    return np.hstack([before, internal_means, after])


def bin_edges_to_centres(edges: np.ndarray) -> np.ndarray:
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


def ternary_grid(
    data: np.ndarray | None = None,
    nbins: int = 10,
    margin: float = 0.001,
    force_margin: bool = False,
    yscale: float = 1.0,
    tfm: Callable = lambda x: x,
) -> tuple[
    np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    """
    Construct a graphical linearly-spaced grid within a ternary space.

    Parameters
    ------------
    data : numpy.ndarray
        Data to construct the grid around (`(samples, 3)`).
    nbins : int
        Number of bins for grid.
    margin : float
        Proportional value for the position of the outer boundary of the grid.
    forge_margin : bool
        Whether to enforce the grid margin.
    yscale : float
        Y scale for the specific ternary diagram.
    tfm :
        Log transform to use for the grid creation.

    Returns
    --------
    bins : numpy.ndarray
        Bin centres along each of the ternary axes (`(samples, 3)`)
    binedges : numpy.ndarray
        Position of bin edges.
    centregrid : list of numpy.ndarray
        Meshgrid of bin centres.
    edgegrid : list of numpy.ndarray
        Meshgrid of bin edges.
    """
    if data is not None:
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
    xbounds, ybounds = ABC_to_xy(bounds, yscale=yscale).T  # in the cartesian xy space
    xbounds = np.hstack((xbounds, [xbounds[0]]))
    ybounds = np.hstack((ybounds, [ybounds[0]]))
    tck, _u = scipy.interpolate.splprep([xbounds, ybounds], per=True, s=0, k=1)
    # interpolated outer boundary
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
