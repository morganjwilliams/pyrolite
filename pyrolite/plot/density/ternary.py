import inspect
import numpy as np
import matplotlib.tri
from ...comp.codata import close, inverse_ilr, ilr, alr, inverse_alr
from ...util.math import flattengrid
from ...util.distributions import sample_kde
from ...util.plot import axes_to_ternary, bin_centres_to_edges
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def ternary_heatmap(
    data,
    bins=20,
    mode="density",
    transform=ilr,
    inverse_transform=inverse_ilr,
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
        grid_transform=tfm,
        grid_inverse_transform=itfm,
    )
    H[~np.isfinite(H)] = 0
    return coords, H, data


"""
import pandas as pd

from mpltern.ternary.datasets import get_scatter_points
import pyrolite.plot


df = pd.DataFrame(np.array([*get_scatter_points(n=500)]).T, columns=["A", "B", "C"])
df = df.loc[df.min(axis=1) > 0.05, :]
cmap = plt.cm.get_cmap("viridis")
cmap.set_under((0, 0, 0, 1))
fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection="ternary"), figsize=(10, 4))
for ix, mode in enumerate(["density", "hist"]):
    coords, H, data = ternary_heatmap(
        df.values,
        bins=20,
        mode=mode,
        remove_background=True,
        transform=alr,
        inverse_transform=inverse_alr,
        grid_border_frac=0.05,
    )
    ax[ix].tripcolor(*coords.T, H.flatten(), cmap=cmap)
plt.tight_layout()

# %%

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax = ax.flat
ax = axes_to_ternary([ax[0], ax[-1]])

ax[0].set_title("data", y=1.2)
ax[1].set_title("transformed data", y=1.2)
ax[2].set_title("ternary grid", y=1.2)

ax[0].scatter(*arr.T, c="k", alpha=0.1)
ax[0].scatter(*ternbound_points.T, c="k")
ax[1].scatter(*ternbound_points_tfmd.T, c="k")
ax[1].scatter(*tdata.T, c="k", alpha=0.1)

ax[2].scatter(*flattengrid(tfm_centregrid).T, c="0.5", marker="x", s=2)
ax[2].scatter(*flattengrid(tfm_edgegrid).T, c="k", marker="x", s=2)
ax[2].pcolormesh(*tfm_edgegrid, H)
ax[2].scatter(*tdata.T, c="white", alpha=0.8, s=4)

ax[-1].scatter(*tern_centre_grid.T, c="k", marker="x", s=2)
ax[-1].tripcolor(*coords.T, H.flatten())
ax[-1].scatter(*arr.T, c="r")
ax[-1].scatter(*ternbound_points.T, c="k")

"""
