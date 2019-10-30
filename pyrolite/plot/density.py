import matplotlib.pyplot as plt
import numpy as np
import logging

from .tern import ternary
from ..comp.codata import close
from ..util.math import on_finite, linspc_, logspc_, linrng_, logrng_, flattengrid
from ..util.distributions import sample_kde
from ..util.plot import (
    ternary_heatmap,
    add_colorbar,
    plot_Z_percentiles,
    percentile_contour_values_from_meshz,
    bin_centres_to_edges,
    __DEFAULT_CONT_COLORMAP__,
    __DEFAULT_DISC_COLORMAP__,
)
from ..util.meta import get_additional_params, subkwargs

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


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
    figsize=(6, 6),
    cmap=__DEFAULT_CONT_COLORMAP__,
    vmin=0.0,
    shading="flat",
    colorbar=False,
    pcolor=False,
    no_ticks=False,
    **kwargs
):
    """
    Creates diagramatic representation of data density and/or frequency for either
    binary diagrams (X-Y) or in a ternary plot
    (limited functionality and poorly tested for the latter).
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
    coverage_scale : :class:`float`, 1.1
        Scale the area over which the density plot is drawn.
    contours : :class:`list`
        Contours to add to the plot.
    percentiles :  :class:`bool`, `True`
        Whether contours specified are to be converted to percentiles.
    relim : :class:`bool`, :code:`True`
        Whether to relimit the plot based on xmin, xmax values.
    figsize : :class:`tuple`, (6, 6)
        Size of the figure generated.
    cmap : :class:`matplotlib.colors.Colormap`
        Colormap for mapping surfaces.
    vmin : :class:`float`, 0.
        Minimum value for colormap.
    shading : :class:`str`, 'flat'
        Shading to apply to pcolormesh.
    colorbar : :class:`bool`, False
        Whether to append a linked colorbar to the generated mappable image.
    pcolor : :class:`bool`
        Option to use the :func:`matplotlib.pyplot.pcolor` function in place
        of :func:`matplotlib.pyplot.pcolormesh`.
    no_ticks : :class:`bool`
        Option to *suppress* tickmarks and labels.

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
            :func:`pyrolite.plot.tern.ternary`
    """
    if (mode == "density") & np.isclose(vmin, 0.0):  # if vmin is not specified
        vmin = 0.02  # 2% max height | 98th percentile

    ax = ax or plt.subplots(1, figsize=figsize)[1]

    background_color = (*ax.patch.get_facecolor()[:-1], 0.0)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap.set_under(color=(1, 1, 1, 0))

    if mode == "density":
        cbarlabel = "Kernel Density Estimate"
    else:
        cbarlabel = "Frequency"

    if pcolor:
        pc = ax.pcolor
    else:
        pc = ax.pcolormesh

    exp = (coverage_scale - 1.0) / 2
    valid_rows = np.isfinite(arr).all(axis=-1)
    if valid_rows.any():
        # Data can't be plotted if there's any nans, so we can exclude these
        arr = arr[valid_rows]

        if arr.shape[-1] == 2:  # binary
            x, y = arr.T
            if extent is not None:  # Expanded extent
                xmin, xmax, ymin, ymax = extent
            else:
                # get the range from the data itself. data > 0 for log grids
                xmin, xmax = [linrng_, logrng_][logx](x, exp=exp)
                ymin, ymax = [linrng_, logrng_][logy](y, exp=exp)

            xstep = [(xmax - xmin) / bins, (xmax / xmin) / bins][logx]
            ystep = [(ymax - ymin) / bins, (ymax / ymin) / bins][logy]
            extent = xmin, xmax, ymin, ymax

            if mode == "hexbin":
                hex_extent = (
                    [
                        [xmin - xstep, xmax + xstep],
                        [np.log(xmin / xstep), np.log(xmax * xstep)],
                    ][logx]
                    + [
                        [ymin - ystep, ymax + ystep],
                        [np.log(ymin / ystep), np.log(ymax * ystep)],
                    ][logy]
                )
                # extent values are exponents (i.e. 3 -> 10**3)
                mappable = ax.hexbin(
                    x,
                    y,
                    gridsize=bins,
                    cmap=cmap,
                    extent=hex_extent,
                    xscale=["linear", "log"][logx],
                    yscale=["linear", "log"][logy],
                    **kwargs
                )

            elif mode == "hist2d":
                if logx:
                    assert (xmin / xstep) > 0.0
                if logy:
                    assert (ymin / ystep) > 0.0

                xs = [linspc_, logspc_][logx](xmin, xmax, xstep, bins)
                ys = [linspc_, logspc_][logy](ymin, ymax, ystep, bins)
                xi, yi = np.meshgrid(xs, ys)  # indexes for potential contouring..
                xe, ye = (
                    bin_centres_to_edges(np.sort(xs)),
                    bin_centres_to_edges(np.sort(ys)),
                )
                range = [[extent[0], extent[1]], [extent[2], extent[3]]]
                zi, xe, ye, im = ax.hist2d(
                    x, y, bins=[xe, ye], range=range, cmap=cmap, **kwargs
                )
                mappable = im

            elif mode == "density":
                if logx:
                    assert xmin > 0.0
                if logy:
                    assert ymin > 0.0

                # Generate Grid of centres
                xs = [linspc_, logspc_][logx](xmin, xmax, bins=bins)
                ys = [linspc_, logspc_][logy](ymin, ymax, bins=bins)
                xe, ye = bin_centres_to_edges(xs), bin_centres_to_edges(ys)

                assert np.isfinite(xs).all() and np.isfinite(ys).all()
                kdedata = arr
                if logx:  # generate x grid over range spanned by log(x)
                    kdedata[:, 0] = np.log(kdedata[:, 0])
                    xs = np.log(xs)
                    xe = np.log(xe)
                if logy:  # generate y grid over range spanned by log(y)
                    kdedata[:, 1] = np.log(kdedata[:, 1])
                    ys = np.log(ys)
                    ye = np.log(ye)

                xymesh = np.meshgrid(xs, ys)
                xi, yi = xymesh

                zi = sample_kde(kdedata, flattengrid(xymesh))
                zi = zi.reshape(xi.shape)
                if percentiles:  # 98th percentile
                    vmin = percentile_contour_values_from_meshz(zi, [1.0 - vmin])[1][0]
                    logger.debug(
                        "Updating `vmin` to percentile equiv: {:.2f}".format(vmin)
                    )
                if logx:
                    xi = np.exp(xi)
                    xe = np.exp(xe)
                if logy:
                    yi = np.exp(yi)
                    ye = np.exp(ye)

                xe, ye = np.meshgrid(xe, ye)

                if not contours:
                    # pcolormesh using bin edges
                    mappable = pc(
                        xe,
                        ye,
                        zi,
                        cmap=cmap,
                        shading=shading,
                        vmin=vmin,
                        **subkwargs(kwargs, pc)
                    )
                    mappable.set_edgecolor(background_color)
                    mappable.set_linestyle("None")
                    mappable.set_lw(0.0)

            if relim:
                ax.axis(extent)

        elif arr.shape[-1] == 3:  # ternary
            arr = close(arr)
            scale = kwargs.pop("scale", 100.0)
            aspect = kwargs.pop("aspect", "eq")
            nanarr = np.ones(3) * np.nan  # update to array method
            ternary(nanarr, ax=ax, scale=scale, figsize=figsize, no_ticks=no_ticks)
            tax = ax.tax
            xe, ye, zi, centres = ternary_heatmap(
                arr, bins=bins, mode=mode, aspect=aspect, ret_centres=True
            )
            xi, yi = centres  # coordinates of grid centres for possible contouring
            xi, yi = xi * scale, yi * scale
            zi[np.isnan(zi)] = 0.0
            if percentiles:  # 98th percentile
                vmin = percentile_contour_values_from_meshz(zi, [1.0 - vmin])[1][0]
                logger.debug("Updating `vmin` to percentile equiv: {:.2f}".format(vmin))

            if not contours:
                zi[zi == 0.0] = np.nan  #
                mappable = pc(
                    xe * scale,
                    ye * scale,
                    zi,
                    cmap=cmap,
                    vmin=vmin,
                    **subkwargs(kwargs, pc)
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_aspect("equal")
        else:
            if not arr.ndim in [0, 1, 2]:
                raise NotImplementedError

        if contours:  # could do this in logspace for accuracy?
            levels = contours or kwargs.pop("levels", None)
            cags = xi, yi, zi  # contour-like function arguments, point estimates
            if percentiles and not isinstance(levels, int):
                _cs = plot_Z_percentiles(
                    *cags, ax=ax, percentiles=levels, extent=extent, cmap=cmap, **kwargs
                )
                mappable = _cs
            else:
                if levels is None:
                    levels = MaxNLocator(nbins=10).tick_values(zi.min(), zi.max())
                elif isinstance(levels, int):
                    levels = MaxNLocator(nbins=levels).tick_values(zi.min(), zi.max())
                # filled contours
                mappable = ax.contourf(
                    *cags, extent=extent, levels=levels, cmap=cmap, vmin=vmin, **kwargs
                )
                # contours
                ax.contour(
                    *cags, extent=extent, levels=levels, cmap=cmap, vmin=vmin, **kwargs
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
