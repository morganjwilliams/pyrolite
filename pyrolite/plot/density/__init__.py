"""

Attributes
----------
USE_PCOLOR : :class:`bool`
    Option to use the :func:`matplotlib.pyplot.pcolor` function in place
    of :func:`matplotlib.pyplot.pcolormesh`.
"""
import matplotlib.pyplot as plt
import numpy as np
import logging

from ...comp.codata import close
from ...util.math import on_finite, linspc_, logspc_, linrng_, logrng_, flattengrid
from ...util.distributions import sample_kde
from ...util.plot import (
    ternary_heatmap,
    xy_to_ABC,
    add_colorbar,
    plot_Z_percentiles,
    percentile_contour_values_from_meshz,
    bin_centres_to_edges,
    __DEFAULT_CONT_COLORMAP__,
    __DEFAULT_DISC_COLORMAP__,
    init_axes,
)
from ...util.meta import get_additional_params, subkwargs

from .grid import DensityGrid

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

USE_PCOLOR = False


def _get_density_methods(ax):
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


def density(
    arr,
    ax=None,
    logx=False,
    logy=False,
    bins=20,
    mode="density",
    extent=None,
    coverage_scale=1.1,
    contours=[],
    percentiles=True,
    relim=True,
    cmap=__DEFAULT_CONT_COLORMAP__,
    shading="flat",
    vmin=0.0,
    colorbar=False,
    **kwargs
):
    """
    Creates diagramatic representation of data density and/or frequency for either
    binary diagrams (X-Y) or ternary plots.
    Additional arguments are typically forwarded
    to respective :mod:`matplotlib` functions
    :func:`~matplotlib.pyplot.pcolormesh`,
    :func:`~matplotlib.pyplot.hist2d`,
    :func:`~matplotlib.pyplot.hexbin`,
    :func:`~matplotlib.pyplot.contour`, and
    :func:`~matplotlib.pyplot.contourf` (see Other Parameters, below).

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        Dataframe from which to draw data.
    ax : :class:`matplotlib.axes.Axes`, `None`
        The subplot to draw on.
    logx : :class:`bool`, `False`
        Whether to use a logspaced *grid* on the x axis. Values strictly >0 required.
    logy : :class:`bool`, `False`
        Whether to use a logspaced *grid* on the y axis. Values strictly >0 required.
    bins : :class:`int`, 20
        Number of bins used in the gridded functions (histograms, KDE evaluation grid).
    mode : :class:`str`, 'density'
        Different modes used here: ['density', 'hexbin', 'hist2d']
    extent : :class:`list`
        Predetermined extent of the grid for which to from the histogram/KDE. In the
        general form (xmin, xmax, ymin, ymax).
    contours : :class:`list`
        Contours to add to the plot.
    percentiles :  :class:`bool`, `True`
        Whether contours specified are to be converted to percentiles.
    relim : :class:`bool`, :code:`True`
        Whether to relimit the plot based on xmin, xmax values.
    cmap : :class:`matplotlib.colors.Colormap`
        Colormap for mapping surfaces.
    vmin : :class:`float`, 0.
        Minimum value for colormap.
    shading : :class:`str`, 'flat'
        Shading to apply to pcolormesh.
    colorbar : :class:`bool`, False
        Whether to append a linked colorbar to the generated mappable image.

    {otherparams}

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the densityplot is plotted.

    Todo
    -----
        * More accurate ternary density plots see :func:`~pyrolite.util.plot.ternary_heatmap` for now.
        * Fix the pcolormesh grid - coordinates are corners, need to increase to N+1 pt
        * Contouring in logspace and transformation back to compositional space

    .. seealso::

        Functions:

            :func:`matplotlib.pyplot.pcolormesh`
            :func:`matplotlib.pyplot.hist2d`
            :func:`matplotlib.pyplot.contourf`
    """
    if (mode == "density") & np.isclose(vmin, 0.0):  # if vmin is not specified
        vmin = 0.02  # 2% max height | 98th percentile

    if arr.shape[-1] == 3:
        projection = "ternary"
    else:
        projection = None

    ax = init_axes(ax=ax, projection=projection, **kwargs)

    pcolor, contour, contourf = _get_density_methods(ax)
    background_color = (*ax.patch.get_facecolor()[:-1], 0.0)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    cmap.set_under(color=(1, 1, 1, 0))

    if mode == "density":
        cbarlabel = "Kernel Density Estimate"
    else:
        cbarlabel = "Frequency"

    valid_rows = np.isfinite(arr).all(axis=-1)
    if (arr.size > 0) and valid_rows.any():
        # Data can't be plotted if there's any nans, so we can exclude these
        arr = arr[valid_rows]

        if projection is None:  # binary
            x, y = arr.T
            grid = DensityGrid(x, y, bins=bins, logx=logx, logy=logy, extent=extent)
            xs, ys = grid.grid_xc, grid.grid_yc
            xci, yci = grid.grid_xci, grid.grid_yci
            xe, ye = grid.grid_xe, grid.grid_ye
            xei, yei = grid.grid_xei, grid.grid_yei
            if mode == "hexbin":
                # extent values are exponents (i.e. 3 -> 10**3)
                mappable = ax.hexbin(
                    x,
                    y,
                    gridsize=bins,
                    cmap=cmap,
                    extent=grid.get_hex_extent(),
                    xscale=["linear", "log"][logx],
                    yscale=["linear", "log"][logy],
                    **kwargs
                )

            elif mode == "hist2d":
                zi, xe, ye, im = ax.hist2d(
                    x, y, bins=[xe, ye], range=grid.get_range(), cmap=cmap, **kwargs
                )
                mappable = im

            elif mode == "density":

                zi = grid.kdefrom(
                    arr,
                    xtransform=[lambda x: x, np.log][logx],
                    ytransform=[lambda y: y, np.log][logy],
                )
                if percentiles:  # 98th percentile
                    vmin = percentile_contour_values_from_meshz(zi, [1.0 - vmin])[1][0]
                    logger.debug(
                        "Updating `vmin` to percentile equiv: {:.2f}".format(vmin)
                    )

                if not contours:
                    # pcolormesh using bin edges
                    mappable = pcolor(
                        grid.grid_xei,
                        grid.grid_yei,
                        zi,
                        cmap=cmap,
                        vmin=vmin,
                        shading=shading,
                        **subkwargs(kwargs, pcolor)
                    )
                    mappable.set_edgecolor(background_color)
                    mappable.set_linestyle("None")
                    mappable.set_lw(0.0)

            if relim and (extent is not None):
                ax.axis(extent)
        elif projection == "ternary":  # ternary
            arr = close(arr)
            scale = kwargs.pop("scale", 100.0)
            aspect = kwargs.pop("aspect", "eq")

            # density, histogram etc parsed here
            xe, ye, zi, centres = ternary_heatmap(
                arr,
                bins=bins,
                mode=mode,
                aspect=aspect,
                remove_background=True,
                ret_centres=True,
            )
            xi, yi = centres  # coordinates of grid centres for possible contouring
            xi, yi = xi * scale, yi * scale
            mask = np.isfinite(zi.flatten())
            xi, yi, zi = xi.flatten()[mask], yi.flatten()[mask], zi.flatten()[mask]
            if percentiles:  # 98th percentile
                vmin = percentile_contour_values_from_meshz(zi, [1.0 - vmin])[1][0]
                logger.debug("Updating `vmin` to percentile equiv: {:.2f}".format(vmin))
            if not contours:
                mappable = pcolor(
                    *xy_to_ABC(np.vstack([xi, yi]).T / scale).T,
                    zi,
                    cmap=cmap,
                    vmin=vmin,
                    shading=shading,
                    **subkwargs(kwargs, pcolor)
                )
            ax.set_aspect("equal")
        else:
            if not arr.ndim in [0, 1, 2]:
                raise NotImplementedError

        if contours:  # could do this in logspace for accuracy?
            mappable = _add_contours(
                grid.grid_xci,
                grid.grid_yci,
                zi=zi.reshape(grid.grid_xci.shape),
                ax=ax,
                contours=contours,
                percentiles=percentiles,
                cmap=cmap,
                vmin=vmin,
                **kwargs
            )

        if colorbar:
            cbkwargs = kwargs.copy()
            cbkwargs["label"] = cbarlabel
            add_colorbar(mappable, **cbkwargs)

        if relim:
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
    return ax


def _add_contours(
    *coords,
    zi=None,
    ax=None,
    contours=[],
    percentiles=True,
    cmap=__DEFAULT_CONT_COLORMAP__,
    vmin=0.0,
    extent=None,
    **kwargs
):
    # get the contour levels
    levels = contours or kwargs.get("levels", None)

    if percentiles and not isinstance(levels, int):
        # plot individual percentile contours
        _cs = plot_Z_percentiles(
            *coords, zi, ax=ax, percentiles=levels, extent=extent, cmap=cmap, **kwargs
        )
        mappable = _cs
    else:
        # plot interval contours
        if levels is None:
            levels = MaxNLocator(nbins=10).tick_values(zi.min(), zi.max())
        elif isinstance(levels, int):
            levels = MaxNLocator(nbins=levels).tick_values(zi.min(), zi.max())
        else:
            raise NotImplementedError
        # filled contours
        mappable = contourf(
            coords, zi, extent=extent, levels=levels, cmap=cmap, vmin=vmin, **kwargs
        )
        # contours
        contour(
            xi, yi, zi, extent=extent, levels=levels, cmap=cmap, vmin=vmin, **kwargs
        )
    return mappable


_add_additional_parameters = True

density.__doc__ = density.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            density,
            plt.pcolormesh,
            plt.hist2d,
            plt.hexbin,
            plt.contour,
            plt.contourf,
            header="Other Parameters",
            indent=4,
            subsections=True,
        ),
    ][_add_additional_parameters]
)
