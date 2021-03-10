import inspect
import numpy as np
import matplotlib.tri
from ...comp.codata import close, inverse_ILR, ILR, ALR, inverse_ALR
from ...util.math import flattengrid
from ...util.distributions import sample_kde
from ...util.plot.grid import bin_centres_to_edges
from ...util.log import Handle

logger = Handle(__name__)


def ternary_heatmap(
    data,
    bins=20,
    mode="density",
    transform=ILR,
    inverse_transform=inverse_ILR,
    ternary_min_value=0.0001,  # 0.01%
    grid_border_frac=0.1,  # 110% range for grid
    grid=None,
    **kwargs
):
    """
    Heatmap for ternary diagrams. This invokes a 3D to 2D transform such as a
    log transform prior to creating a grid.

    Parameters
    -----------
    data : :class:`numpy.ndarray`
        Ternary data to obtain heatmap coords from.
    bins : :class:`int`
        Number of bins for the grid.
    mode : :class:`str`, :code:`{'histogram', 'density'}`
        Which mode to render the histogram/KDE in.
    transform : :class:`callable` | :class:`sklearn.base.TransformerMixin`
        Callable function or Transformer class.
    inverse_transform : :class:`callable`
        Inverse function for `transform`, necessary if transformer class not specified.
    ternary_min_value : :class:`float`
        Optional specification of minimum values within a ternary diagram to draw the
        transformed grid.
    grid_border_frac : :class:`float`
        Size of border around the grid, expressed as a fraction of the total grid range.
    grid : :class:`numpy.ndarray`
        Grid coordinates to sample at, if already calculated. For the density mode,
        this is a (nsamples, 2) array. For histograms, this is a two-member list of
        bin edges.

    Returns
    -------
    t, l, r : :class:`tuple` of :class:`numpy.ndarray`
        Ternary coordinates for the heatmap.
    H : :class:`numpy.ndarray`
        Histogram/density estimates for the coordinates.
    data : :class:`dict`
        Data dictonary with grid arrays and relevant information.

    Notes
    -----

    Zeros will not render in this heatmap, consider replacing zeros with small values
    or imputing them if they must be incorporated.
    """
    arr = close(data)  # should remove zeros/nans
    arr = arr[np.isfinite(arr).all(axis=1)]

    if inspect.isclass(transform):
        # TransformerMixin
        tcls = transform()
        tfm = tcls.transform
        itfm = tcls.inverse_transform
    else:
        # callable
        tfm = transform
        assert callable(inverse_transform)
        itfm = inverse_transform

    if grid is None:
        # minimum bounds on triangle
        mins = np.max(
            np.vstack([np.ones(3) * ternary_min_value, np.nanmin(arr, axis=0)]), axis=0
        )
        maxs = np.min(
            np.vstack([1 - np.ones(3) * ternary_min_value, np.nanmax(arr, axis=0)]),
            axis=0,
        )

        # let's construct a bounding triangle
        # three points defining the edges of what will be rendered
        ternbounds = np.vstack([mins, maxs]).T
        ternbound_points = np.array(
            [
                [
                    (1 - np.sum([ternbounds[i, 0] for z in range(3) if z != ix]))
                    if i == ix
                    else ternbounds[i, 0]
                    for i in range(3)
                ]
                for ix in range(3)
            ]
        )
        ternbound_points_tfmd = tfm(ternbound_points)

        tdata = tfm(arr)
        tfm_min, tfm_max = np.min(tdata, axis=0), np.max(tdata, axis=0)
        trng = tfm_max - tfm_min
        brdr = grid_border_frac / 2
        tfm_bin_centres = [
            np.linspace(
                np.nanmin(tdata[:, dim]) - brdr * trng[dim],  # small step back
                np.nanmax(tdata[:, dim]) + brdr * trng[dim],  # small step forward
                bins,
            )
            for dim in range(2)
        ]
        tfm_bin_edges = [bin_centres_to_edges(b) for b in tfm_bin_centres]

        tfm_centregrid = np.meshgrid(*tfm_bin_centres)
        tfm_edgegrid = np.meshgrid(*tfm_bin_edges)

        tern_centre_grid = itfm(flattengrid(tfm_centregrid))
        tern_edge_grid = itfm(flattengrid(tfm_edgegrid))

    if mode == "density":
        dgrid = grid or flattengrid(tfm_edgegrid)
        H = sample_kde(tdata, dgrid)
        H = H.reshape(tfm_edgegrid[0].shape)
        coords = tern_edge_grid
    elif "hist" in mode:
        hgrid = grid or tfm_bin_edges
        H, hedges = np.histogramdd(tdata, bins=hgrid)
        H = H.T
        coords = tern_centre_grid
    elif "hex" in mode:
        raise NotImplementedError
    else:
        raise NotImplementedError

    data = dict(
        tfm_centres=tfm_centregrid,
        tfm_edges=tfm_edgegrid,
        tern_edges=tern_edge_grid,
        tern_centres=tern_centre_grid,
        tern_bound_points=ternbound_points,
        tfm_tern_bound_points=ternbound_points_tfmd,
        grid_transform=tfm,
        grid_inverse_transform=itfm,
    )
    H[~np.isfinite(H)] = 0
    return coords, H, data
