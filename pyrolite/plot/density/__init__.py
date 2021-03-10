"""
Kernel desnity estimation plots for geochemical data.
"""
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from ...comp.codata import close
from ...util.plot.density import (
    plot_Z_percentiles,
    percentile_contour_values_from_meshz,
    get_axis_density_methods,
)
from ...util.plot.style import DEFAULT_CONT_COLORMAP
from ...util.plot.axes import init_axes, add_colorbar
from ...util.meta import get_additional_params, subkwargs
from .grid import DensityGrid
from .ternary import ternary_heatmap
from ...util.log import Handle

logger = Handle(__name__)


def density(
    arr,
    ax=None,
    logx=False,
    logy=False,
    bins=25,
    mode="density",
    extent=None,
    contours=[],
    percentiles=True,
    relim=True,
    cmap=DEFAULT_CONT_COLORMAP,
    shading="auto",
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
        Contours to add to the plot, where :code:`mode='density'` is used.
    percentiles :  :class:`bool`, `True`
        Whether contours specified are to be converted to percentiles.
    relim : :class:`bool`, :code:`True`
        Whether to relimit the plot based on xmin, xmax values.
    cmap : :class:`matplotlib.colors.Colormap`
        Colormap for mapping surfaces.
    vmin : :class:`float`, 0.
        Minimum value for colormap.
    shading : :class:`str`, 'auto'
        Shading to apply to pcolormesh.
    colorbar : :class:`bool`, False
        Whether to append a linked colorbar to the generated mappable image.

    {otherparams}

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the densityplot is plotted.

    .. seealso::

        Functions:

            :func:`matplotlib.pyplot.pcolormesh`
            :func:`matplotlib.pyplot.hist2d`
            :func:`matplotlib.pyplot.contourf`

    Notes
    -----
    Could implement an option and filter to 'scatter' points below the minimum threshold
    or maximum percentile contours.
    """
    if (mode == "density") & np.isclose(vmin, 0.0):  # if vmin is not specified
        vmin = 0.02  # 2% max height | 98th percentile

    if arr.shape[-1] == 3:
        projection = "ternary"
    else:
        projection = None

    ax = init_axes(ax=ax, projection=projection, **kwargs)

    pcolor, contour, contourf = get_axis_density_methods(ax)
    background_color = (*ax.patch.get_facecolor()[:-1], 0.0)

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        cmap = copy.copy(cmap)  # without this, it would modify the global cmap
        cmap.set_under((1, 1, 1, 0))

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
            grid = DensityGrid(
                x,
                y,
                bins=bins,
                logx=logx,
                logy=logy,
                extent=extent,
                **subkwargs(kwargs, DensityGrid)
            )
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
                    **subkwargs(kwargs, ax.hexbin)
                )

            elif mode == "hist2d":
                zi, xe, ye, im = ax.hist2d(
                    x,
                    y,
                    bins=[grid.grid_xe, grid.grid_ye],
                    range=grid.get_range(),
                    cmap=cmap,
                    cmin=[0, 1][vmin > 0],
                    **subkwargs(kwargs, ax.hist2d)
                )
                mappable = im

            elif mode == "density":
                zei = grid.kdefrom(
                    arr,
                    xtransform=[lambda x: x, np.log][logx],
                    ytransform=[lambda y: y, np.log][logy],
                    mode="edges",
                    **subkwargs(kwargs, grid.kdefrom)
                )

                if percentiles:  # 98th percentile
                    vmin = percentile_contour_values_from_meshz(zei, [1.0 - vmin])[1][0]
                    logger.debug(
                        "Updating `vmin` to percentile equiv: {:.2f}".format(vmin)
                    )

                if not contours:
                    # pcolormesh using bin edges
                    mappable = pcolor(
                        grid.grid_xei,
                        grid.grid_yei,
                        zei,
                        cmap=cmap,
                        vmin=vmin,
                        shading=shading,
                        **subkwargs(kwargs, pcolor)
                    )
                    mappable.set_edgecolor(background_color)
                    mappable.set_linestyle("None")
                    mappable.set_lw(0.0)
                else:
                    mappable = _add_contours(
                        grid.grid_xei,
                        grid.grid_yei,
                        zi=zei.reshape(grid.grid_xei.shape),
                        ax=ax,
                        contours=contours,
                        percentiles=percentiles,
                        cmap=cmap,
                        vmin=vmin,
                        **kwargs
                    )
            if relim and (extent is not None):
                ax.axis(extent)
        elif projection == "ternary":  # ternary
            if shading == 'auto':
                shading = 'flat' # auto cant' be passed to tripcolor
            # zeros make nans in this case, due to the heatmap calculations
            arr[~(arr > 0).all(axis=1), :] = np.nan
            arr = close(arr)
            if mode == "hexbin":
                raise NotImplementedError
            # density, histogram etc parsed here
            coords, zi, data = ternary_heatmap(arr, bins=bins, mode=mode)

            if percentiles:  # 98th percentile
                vmin = percentile_contour_values_from_meshz(zi, [1.0 - vmin])[1][0]
                logger.debug("Updating `vmin` to percentile equiv: {:.2f}".format(vmin))

            # remove coords where H==0, as ax.tripcolor can't deal with variable alpha :'(
            fltr = (zi != 0) & (zi >= vmin)
            coords = coords[fltr.flatten(), :]
            zi = zi[fltr]

            if not contours:
                tri_poly_collection = pcolor(
                    *coords.T,
                    zi.flatten(),
                    cmap=cmap,
                    vmin=vmin,
                    shading=shading,
                    **subkwargs(kwargs, pcolor)
                )

                mappable = tri_poly_collection
            else:
                mappable = _add_contours(
                    *coords.T,
                    zi=zi.flatten(),
                    ax=ax,
                    contours=contours,
                    percentiles=percentiles,
                    cmap=cmap,
                    vmin=vmin,
                    **kwargs
                )
            ax.set_aspect("equal")
        else:
            if not arr.ndim in [0, 1, 2]:
                raise NotImplementedError

        if colorbar:
            cbkwargs = kwargs.copy()
            cbkwargs["label"] = cbarlabel
            add_colorbar(mappable, **cbkwargs)

    return ax


def _add_contours(
    *coords,
    zi=None,
    ax=None,
    contours=[],
    cmap=DEFAULT_CONT_COLORMAP,
    vmin=0.0,
    extent=None,
    **kwargs
):
    """
    Add density-based contours to a plot.
    """
    # get the contour levels
    percentiles = kwargs.pop("percentiles", True)
    levels = contours or kwargs.get("levels", None)
    pcolor, contour, contourf = get_axis_density_methods(ax)
    if percentiles and not isinstance(levels, int):
        # plot individual percentile contours
        _cs = plot_Z_percentiles(
            *coords,
            zi=zi,
            ax=ax,
            percentiles=levels,
            extent=extent,
            cmap=cmap,
            **kwargs
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
            *coords, zi, extent=extent, levels=levels, cmap=cmap, vmin=vmin, **kwargs
        )
        # contours
        contour(
            *coords, zi, extent=extent, levels=levels, cmap=cmap, vmin=vmin, **kwargs
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
