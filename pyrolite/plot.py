import pandas as pd
import pandas_flavor as pf
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import ternary
from .util.pd import to_frame
from .util.math import on_finite
from .util.plot import (
    ABC_to_tern_xy,
    tern_heatmapcoords,
    add_colorbar,
    plot_Z_percentiles,
    percentile_contour_values_from_meshz,
)
from .util.general import iscollection
from .geochem import common_elements, REE, get_ionic_radii
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

DEFAULT_CONT_COLORMAP = plt.cm.viridis
DEFAULT_DISC_COLORMAP = plt.cm.tab10


@pf.register_series_method
@pf.register_dataframe_method
def spiderplot(
    df, components: list = None, ax=None, plot=True, fill=False, indexes=None, **kwargs
):
    """
    Plots spidergrams for trace elements data. Additional keyword arguments are
    passed to matplotlib.

    By using separate lines and scatterplots, values between two missing
    items are still presented. Might be able to speed up the lines
    with a matplotlib.collections.LineCollection.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe from which to draw data.
    components : :class:`list`, None
        Elements or compositional components to plot.
    ax : :class:`matplotlib.axes.Axes`, None
        The subplot to draw on.
    plot : :class:`bool`, True
        Whether to plot lines and markers.
    fill : :class:`bool`, True
        Whether to add a patch representing the full range.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the spiderplot is plotted.
    """
    kwargs = kwargs.copy()
    try:
        assert plot or fill
    except:
        msg = "Please select to either plot values or fill between ranges."
        raise AssertionError(msg)

    df = to_frame(df)

    if components is None:
        components = [el for el in common_elements() if el in df.columns]

    assert len(components) != 0

    sty = {}

    # Color ----------------------------------------------------------

    variable_colors = False
    color = kwargs.get("color") or kwargs.get("c")
    norm = kwargs.pop("norm", None)
    if color is not None:
        if iscollection(color):
            sty["c"] = color
            variable_colors = True
        else:
            sty["color"] = color

    _c = sty.pop("c", None)

    cmap = kwargs.get("cmap", None)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if (_c is not None) and (cmap is not None):
        if norm is not None:
            _c = [norm(c) for c in _c]
        _c = [cmap(c) for c in _c]

    sty["alpha"] = kwargs.get("alpha") or kwargs.get("a") or 1.0

    # ---------------------------------------------------------------------

    ax = ax or plt.subplots(1, figsize=(len(components) * 0.3, 4))[1]

    if indexes is None:
        indexes = np.arange(len(components))
        ax.set_xticks(indexes)
        ax.set_xticklabels(components, rotation=60)

    if fill:
        mins = df.loc[:, components].min(axis=0)
        maxs = df.loc[:, components].max(axis=0)
        plycol = ax.fill_between(indexes, mins, maxs, **sty)
        # Use the first (typically only) element for color
        if (sty.get("color") is None) and (sty.get("c") is None):
            sty["color"] = plycol.get_facecolor()[0]

    sty["marker"] = kwargs.get("marker", None) or "D"

    # Use the default color cycling to provide a single color
    if sty.get("color") is None and _c is None:
        sty["color"] = next(ax._get_lines.prop_cycler)["color"]

    if plot:
        ls = ax.plot(indexes, df.loc[:, components].T.values.astype(np.float), **sty)
        if variable_colors:
            for l, c in zip(ls, _c):
                l.set_color(c)

        sty["s"] = kwargs.get("markersize") or kwargs.get("s") or 5.0
        if (sty.get("color") is None) and (_c is None):
            sty["color"] = ls[0].get_color()

        sty["label"] = kwargs.pop("label", None)
        # For the scatter, the number of points > the number of series
        # Need to check if this is the case, and create equivalent

        if _c is not None:
            cshape = np.array(_c).shape
            if cshape != df.loc[:, components].shape:
                # expand it across the columns
                _c = np.tile(_c, (len(components), 1))

        sc = ax.scatter(
            np.tile(indexes, (df.loc[:, components].index.size, 1)).T,
            df.loc[:, components].T.values.astype(np.float),
            **sty,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Element")

    unused_keys = [i for i in kwargs if i not in list(sty.keys())]
    if len(unused_keys):
        logger.info("Styling not yet implemented for:{}".format(unused_keys))

    return ax


@pf.register_series_method
@pf.register_dataframe_method
def REE_radii_plot(df=None, ax=None, **kwargs):
    """
    Creates an axis for a REE diagram with ionic radii along the x axis.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Optional designation of axes to reconfigure.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the REE_radii_plot is plotted.
    """
    if ax is not None:
        fig = ax.figure
        ax = ax
    else:
        fig, ax = plt.subplots()

    ree = REE()
    radii = np.array(get_ionic_radii(ree, charge=3, coordination=8))

    if df is not None:
        if any([i in df.columns for i in ree]):
            reedata = df.loc[:, ree]
            reedata.spiderplot(ax=ax, indexes=radii, **kwargs)

    _ax = ax.twiny()
    ax.set_yscale("log")
    ax.set_xlim((0.99 * radii.min(), 1.01 * radii.max()))
    _ax.set_xlim(ax.get_xlim())
    _ax.set_xticks(radii)
    _ax.set_xticklabels(ree)
    _ax.set_xlabel("Element")
    ax.axhline(1.0, ls="--", c="k", lw=0.5)
    ax.set_ylabel(" $\mathrm{X / X_{Reference}}$")
    ax.set_xlabel("Ionic Radius ($\mathrm{\AA}$)")

    return ax


@pf.register_series_method
@pf.register_dataframe_method
def ternaryplot(df, components: list = None, ax=None, clockwise=True, **kwargs):
    """
    Plots scatter ternary diagrams, using a wrapper around the
    python-ternary library (gh.com/marcharper/python-ternary).
    Additional keyword arguments arepassed to matplotlib.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe from which to draw data.
    components : :class:`list`, None
        Elements or compositional components to plot.
    ax : :class:`matplotlib.axes.Axes`, None
        The subplot to draw on.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the spiderplot is plotted.
    """
    kwargs = kwargs.copy()
    df = to_frame(df)

    try:
        if not df.columns.size == 3:
            assert len(components) == 3

        if components is None:
            components = df.columns.values
    except:
        msg = "Suggest components or provide a slice of the dataframe."
        raise AssertionError(msg)

    # Some default values
    scale = kwargs.pop("scale", None) or 100.0
    figsize = kwargs.pop("size", None) or 8.0
    gridsize = kwargs.pop("gridsize", None) or 10.0
    fontsize = kwargs.pop("fontsize", None) or 10.0

    sty = {}
    sty["marker"] = kwargs.pop("marker", None) or "D"
    sty["color"] = kwargs.pop("color", None) or kwargs.pop("c", None) or "0.5"
    sty["label"] = kwargs.pop("label", None)
    sty["alpha"] = kwargs.pop("alpha", None) or kwargs.pop("a", None) or 1.0

    ax = ax or plt.subplots(1, figsize=(figsize, figsize * 3 ** 0.5 * 0.5))[1]
    d1 = ax.__dict__.copy()

    # Checking if there's already a ternary axis
    tax = getattr(ax, "tax", None) or ternary.figure(ax=ax, scale=scale)[1]

    # Set attribute for future reference
    ax.tax = tax
    points = (
        df.loc[:, components].div(df.loc[:, components].sum(axis=1), axis=0).values
        * scale
    )
    if points.any():
        tax.scatter(points, **sty)

    if sty["label"] is not None:
        tax.legend(frameon=False)

    # Check if there's already labels
    if not len(tax._labels.keys()):
        tax.left_axis_label(components[2], fontsize=fontsize)
        tax.bottom_axis_label(components[0], fontsize=fontsize)
        tax.right_axis_label(components[1], fontsize=fontsize)

        tax.gridlines(multiple=gridsize, color="k", alpha=0.5)
        tax.ticks(axis="lbr", linewidth=1, clockwise=clockwise, multiple=gridsize)
        tax.boundary(linewidth=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    return ax


def _linspc(_min, _max, step=0.0, bins=20):
    """
    Linear spaced array, with optional step for grid margins.

    Parameters
    -----------
    _min : :class:`numpy.number`
        Minimum value for spaced range.
    _max : :class:`numpy.number`
        Maximum value for spaced range.
    step : :class:`numpy.number`, 0.0
        Step for expanding at grid edges. Default of 0.0 results in no expansion.
    bins : int
        Number of bins to divide the range (adds one by default).

    Returns
    -------
    :class:`numpy.ndarray`
        Linearly-spaced array.
    """
    return np.linspace(_min - step, _max + step, bins + 1)


def _logspc(_min, _max, step=1.0, bins=20):
    """
    Log spaced array, with optional step for grid margins.

    Parameters
    -----------
    _min : :class:`numpy.number`
        Minimum value for spaced range.
    _max : :class:`numpy.number`
        Maximum value for spaced range.
    step : :class:`numpy.number`, 1.0
        Step for expanding at grid edges. Default of 1.0 results in no expansion.
    bins : int
        Number of bins to divide the range (adds one by default).

    Returns
    -------
    :class:`numpy.ndarray`
        Log-spaced array.
    """
    return np.logspace(np.log(_min / step), np.log(_max * step), bins, base=np.e)


def _logrng(v, exp=0.0):
    """
    Range of a sample, where values <0 are excluded.

    Parameters
    -----------
    v : :class:`list`; list-like
        Array of values to obtain a range from.
    exp : :class:`float`, (0, 1)
        Fractional expansion of the range.

    Returns
    -------
    :class:`tuple`
        Min, max tuple.
    """
    u = v[(v > 0)]  # make sure the range_values are >0
    return _linrng(u, exp=exp)


def _linrng(v, exp=0.0):
    """
    Range of a sample, where values <0 are included.

    Parameters
    -----------
    v : :class:`list`; list-like
        Array of values to obtain a range from.
    exp : :class:`float`, (0, 1)
        Fractional expansion of the range.

    Returns
    -------
    :class:`tuple`
        Min, max tuple.
    """
    u = v[np.isfinite(v)]
    return (np.min(u) * (1.0 - exp), np.max(u) * (1.0 + exp))


@pf.register_series_method
@pf.register_dataframe_method
def densityplot(
    df,
    components: list = None,
    ax=None,
    logx=False,
    logy=False,
    mode="density",
    extent=None,
    coverage_scale=1.1,
    contours=[],
    percentiles=True,
    relim=True,
    axlabels=True,
    **kwargs
):
    """
    Plots density plot diagrams. Should work for either binary components (X-Y)
    or in a ternary plot. Some additional keyword arguments are passed to
    matplotlib.

    Todo:
        Split logscales for x and y (currently only for log-log)

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe from which to draw data.
    components : :class:`list`, None
        Elements or compositional components to plot.
    ax : :class:`matplotlib.axes.Axes`, None
        The subplot to draw on.
    logx : :class:`bool`, False
        Whether to use a logspaced *grid* on the x axis. Values strictly >0 required.
    logy : :class:`bool`, False
        Whether to use a logspaced *grid* on the y axis. Values strictly >0 required.
    mode : :class:`str`, 'density'
        Different modes used here: ['density', 'hexbin', 'hist2d']
    extent : :class:`list`; list-like
        Predetermined extent of the grid for which to from the histogram/KDE. In the
        general from (xmin, xmax, ymin, ymax).
    coverage_scale : :class:`float`, 1.1
        Scale the area over which the density plot is drawn.
    contours : :class:`list`
        Contours to add to the plot.
    percentiles :  :class:`bool`, True
        Whether contours specified are to be converted to percentiles.
    relim : :class:`bool`, True
        Whether to relimit the plot based on xmin, xmax values.
    axlabels : :class:`bool`, True
        Whether to add x-y axis labels.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the densityplot is plotted.
    """
    kwargs = kwargs.copy()
    df = to_frame(df)
    try:
        if df.columns.size not in [2, 3]:
            assert len(components) in [2, 3]

        if components is None:
            components = df.columns.values
    except:
        msg = "Suggest components or provide a slice of the dataframe."
        raise AssertionError(msg)

    figsize = kwargs.pop("figsize", 8.0)
    fontsize = kwargs.pop("fontsize", 12.0)
    lws = kwargs.pop("linewidths", None)
    lss = kwargs.pop("linestyles", None)

    colorbar = kwargs.pop("colorbar", False)

    ax = ax or plt.subplots(1, figsize=(figsize, figsize * 3 ** 0.5 * 0.5))[1]
    background_color = ax.patch.get_facecolor()  # consider alpha here
    nbins = kwargs.pop("bins", 20)
    cmap = kwargs.pop("cmap", DEFAULT_CONT_COLORMAP)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap.set_under(color=background_color)

    exp = (coverage_scale - 1.0) / 2
    data = df.loc[:, components].values

    if data.any():
        # Data can't be plotted if there's any nans, so we can exclude these
        data = data[(np.isfinite(data).all(axis=1)), :]

        if len(components) == 2:  # binary
            x, y = data.T
            if extent is not None:  # Expanded extent
                xmin, xmax, ymin, ymax = extent
            else:
                # get the range from the data itself. data > 0 for log grids
                xmin, xmax = [_linrng, _logrng][logx](x, exp=exp)
                ymin, ymax = [_linrng, _logrng][logy](y, exp=exp)

            xstep = [(xmax - xmin) / nbins, (xmax / xmin) / nbins][logx]
            ystep = [(ymax - ymin) / nbins, (ymax / ymin) / nbins][logy]
            extent = xmin, xmax, ymin, ymax

            if mode == "hexbin":
                vmin = kwargs.pop("vmin", 0)
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
                    gridsize=nbins,
                    cmap=cmap,
                    extent=hex_extent,
                    xscale=["linear", "log"][logx],
                    yscale=["linear", "log"][logy],
                    **kwargs,
                )
                cbarlabel = "Frequency"

            elif mode == "hist2d":
                vmin = kwargs.pop("vmin", 0)
                if logx:
                    assert (xmin / xstep) > 0.0
                if logy:
                    assert (ymin / ystep) > 0.0

                xe = [_linspc, _logspc][logx](xmin, xmax, xstep, nbins)
                ye = [_linspc, _logspc][logy](ymin, ymax, ystep, nbins)

                range = [[extent[0], extent[1]], [extent[2], extent[3]]]
                h, xe, ye, im = ax.hist2d(
                    x, y, bins=[xe, ye], range=range, cmap=cmap, **kwargs
                )
                mappable = im
                cbarlabel = "Frequency"

            elif mode == "density":
                shading = kwargs.pop("shading", None) or "flat"

                if logx:
                    assert xmin > 0.0
                if logy:
                    assert ymin > 0.0

                # Generate Grid
                xs = [_linspc, _logspc][logx](xmin, xmax, bins=nbins)
                ys = [_linspc, _logspc][logy](ymin, ymax, bins=nbins)
                xi, yi = np.meshgrid(xs, ys)
                xi, yi = xi.T, yi.T
                assert np.isfinite(xi).all() and np.isfinite(yi).all()

                kdedata = data.T
                if logx:  # generate x grid over range spanned by log(x)
                    kdedata[0] = np.log(kdedata[0])
                    xi = np.log(xi)
                if logy:  # generate y grid over range spanned by log(y)
                    kdedata[1] = np.log(kdedata[1])
                    yi = np.log(yi)

                # remove nan, inf bearing rows
                kdedata = kdedata[:, np.isfinite(kdedata).all(axis=0)]
                assert np.isfinite(kdedata).all()
                k = gaussian_kde(kdedata)  # gaussian kernel approximation on the grid
                zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                assert np.isfinite(zi).all()
                zi = zi / zi.max()
                vmin = kwargs.pop("vmin", 0.02)  # 2% max height
                if percentiles:  # 98th percentile
                    vmin = percentile_contour_values_from_meshz(zi, [1.0 - vmin])[1][0]
                    logger.debug(
                        "Updating `vmin` to percentile equiv: {:.2f}".format(vmin)
                    )
                if logx:
                    xi = np.exp(xi)
                if logy:
                    yi = np.exp(yi)
                if contours:
                    levels = contours or kwargs.pop("levels", None)
                    cags = xi, yi, zi
                    if percentiles and not isinstance(levels, int):
                        _cs = plot_Z_percentiles(
                            *cags,
                            ax=ax,
                            percentiles=levels,
                            extent=extent,
                            zorder=kwargs.pop('zorder', 5),
                            cmap=cmap,
                            **kwargs,
                        )
                        mappable = _cs
                    else:
                        if levels is None:
                            levels = MaxNLocator(nbins=10).tick_values(
                                zi.min(), zi.max()
                            )
                        elif isinstance(levels, int):
                            levels = MaxNLocator(nbins=levels).tick_values(
                                zi.min(), zi.max()
                            )

                        # filled contours
                        mappable = ax.contourf(
                            *cags, extent=extent, levels=levels, vmin=vmin, **kwargs
                        )
                        # contours
                        ax.contour(
                            *cags,
                            extent=extent,
                            levels=levels,
                            linewidths=lws,
                            linestyles=lss,
                            vmin=vmin,
                            **kwargs
                        )
                else:
                    mappable = ax.pcolormesh(
                        xi, yi, zi, cmap=cmap, shading=shading, vmin=vmin, **kwargs
                    )
                    mappable.set_edgecolor(background_color)
                    mappable.set_linestyle("None")
                    mappable.set_lw(0.0)

                cbarlabel = "Kernel Density Estimate"

            if colorbar:
                cbkwargs = kwargs.copy()
                cbkwargs["label"] = cbarlabel
                add_colorbar(mappable, **cbkwargs)

            if relim:
                ax.axis(extent)
            if axlabels:
                ax.set_xlabel(components[0], fontsize=fontsize)
                ax.set_ylabel(components[1], fontsize=fontsize)

        elif len(components) == 3:  # ternary
            scale = kwargs.pop("scale", None) or 100.0
            empty_df = pd.DataFrame(columns=df.columns)
            heatmapdata = tern_heatmapcoords(data.T, scale=nbins, bins=nbins)
            ternaryplot(empty_df, ax=ax, components=components, scale=scale)
            tax = ax.tax
            if mode == "hexbin":
                style = "hexagonal"
            else:
                style = "triangular"
            tax.heatmap(
                heatmapdata, scale=scale, style=style, colorbar=colorbar, **kwargs
            )
        else:
            pass
    if relim:
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
    plt.tight_layout()
    return ax
