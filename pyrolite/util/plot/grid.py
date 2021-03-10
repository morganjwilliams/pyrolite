"""
Gridding and binning functions.
"""
import numpy as np
from ...comp.codata import close
from .transform import ABC_to_xy, xy_to_ABC
import scipy.interpolate
from ..log import Handle

logger = Handle(__name__)


def bin_centres_to_edges(centres, sort=True):
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


def ternary_grid(
    data=None, nbins=10, margin=0.001, force_margin=False, yscale=1.0, tfm=lambda x: x
):
    """
    Construct a graphical linearly-spaced grid within a ternary space.

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
        Bin centres along each of the ternary axes (:code:`(samples, 3)`)
    binedges : :class:`numpy.ndarray`
        Position of bin edges.
    centregrid : :class:`list` of :class:`numpy.ndarray`
        Meshgrid of bin centres.
    edgegrid : :class:`list` of :class:`numpy.ndarray`
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
    tck, u = scipy.interpolate.splprep([xbounds, ybounds], per=True, s=0, k=1)
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
