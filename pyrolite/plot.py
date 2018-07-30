import pandas as pd
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import ternary
from .util.pd import to_frame
from .util.math import on_finite
from .util.plot import ABC_to_tern_xy, tern_heatmapcoords, add_colorbar
from .geochem import common_elements
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

DEFAULT_CONT_COLORMAP = plt.cm.viridis
DEFAULT_DISC_COLORMAP = 'tab10'


def spiderplot(df, components:list=None, ax=None, plot=True, fill=False, **kwargs):
    """
    Plots spidergrams for trace elements data.
    By using separate lines and scatterplots, values between two null-valued
    items are still presented. Might be able to speed up the lines
    with a matplotlib.collections.LineCollection

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe from which to draw data.
    components: list, None
        Elements or compositional components to plot.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    plot: boolean, True
        Whether to plot lines and markers.
    fill:
        Whether to add a patch representing the full range.
    style:
        Styling keyword arguments to pass to matplotlib.
    """
    kwargs = kwargs.copy()
    try:
        assert plot or fill
    except:
        raise AssertionError('Please select to either plot values or fill between ranges.')

    df = to_frame(df)

    if components is None:
        components = [el for el in common_elements(output='str')
                      if el in df.columns]

    assert len(components) != 0
    c_indexes = np.arange(len(components))

    sty = {}
    # Some default values


    sty['color'] = kwargs.pop('color', None) or kwargs.pop('c', None)
    sty['alpha'] = kwargs.pop('alpha', None) or kwargs.pop('a', None) or 1.
    if sty['color'] is None:
        del sty['color']

    ax = ax or plt.subplots(1, figsize=(len(components)*0.3, 4))[1]

    if fill:
        mins = df.loc[:, components].min(axis=0)
        maxs = df.loc[:, components].max(axis=0)
        plycol = ax.fill_between(c_indexes, mins, maxs, **sty)
        # Use the first (typically only) element for color
        sty['color'] = sty.pop('color', None) or plycol.get_facecolor()[0]

    sty['marker'] = kwargs.pop('marker', None) or 'D'
    if plot:
        # Use the default color cycling to provide a single color
        if 'color' not in sty.keys():
            sty['color'] = next(ax._get_lines.prop_cycler)['color']

        ls = ax.plot(c_indexes,
                     df.loc[:, components].T.values.astype(np.float),
                     **sty)

        sty['s'] = kwargs.get('markersize') or kwargs.get('s') or 5.
        if sty.get('color') is None:
            sty['color'] = ls[0].get_color()

        sty['label'] = kwargs.pop('label', None)
        sc = ax.scatter(np.tile(c_indexes, (df.loc[:, components].index.size,1)).T,
                        df.loc[:, components].T.values.astype(np.float),
                        **sty)

    ax.set_xticks(c_indexes)
    ax.set_xticklabels(components, rotation=60)
    ax.set_yscale('log')
    ax.set_xlabel('Element')

    unused_keys = [i for i in kwargs if i not in list(sty.keys())]
    if len(unused_keys):
        logger.info('Styling not yet implemented for:{}'.format(unused_keys))

    return ax


def ternaryplot(df, components:list=None, ax=None, **kwargs):
    """
    Plots scatter ternary diagrams, using a wrapper around the
    python-ternary library (gh.com/marcharper/python-ternary).

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe from which to draw data.
    components: list, None
        Elements or compositional components to plot.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    """
    kwargs = kwargs.copy()
    df = to_frame(df)

    try:
        if not df.columns.size == 3:
            assert len(components) == 3

        if components is None:
            components = df.columns.values
    except:
        raise AssertionError('Please either suggest three elements or a 3-element dataframe.')

    # Some default values
    scale = kwargs.pop('scale', None) or 100.
    figsize = kwargs.pop('size', None) or 8.
    gridsize = kwargs.pop('gridsize', None) or 10.
    fontsize = kwargs.pop('fontsize', None) or 12.

    sty = {}
    sty['marker'] = kwargs.pop('marker', None) or 'D'
    sty['color'] = kwargs.pop('color', None) or kwargs.pop('c', None) or '0.5'
    sty['label'] = kwargs.pop('label', None)
    sty['alpha'] = kwargs.pop('alpha', None) or kwargs.pop('a', None) or 1.

    ax = ax or plt.subplots(1, figsize=(figsize, figsize* 3**0.5 * 0.5))[1]
    d1 = ax.__dict__.copy()

     # Checking if there's already a ternary axis
    tax = getattr(ax, 'tax', None) or ternary.figure(ax=ax, scale=scale)[1]

    # Set attribute for future reference
    ax.tax = tax
    points = df.loc[:, components].div(df.loc[:, components].sum(axis=1), axis=0).values * scale
    if points.any():
        tax.scatter(points, **sty)

    if sty['label'] is not None:
        tax.legend(frameon=False,)

    # Check if there's already labels
    if not len(tax._labels.keys()):
        tax.left_axis_label(components[2], fontsize=fontsize)
        tax.bottom_axis_label(components[0], fontsize=fontsize)
        tax.right_axis_label(components[1], fontsize=fontsize)

        tax.gridlines(multiple=gridsize, color='k', alpha=0.5)
        tax.ticks(axis='lbr', linewidth=1, multiple=gridsize)
        tax.boundary(linewidth=1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    return tax


def densityplot(df,
                components:list=None,
                ax=None,
                mode='density',
                coverage_scale=1.1,
                logspace=False,
                contour=False,
                **kwargs):
    """
    Plots density plot diagrams.

    Should work for either binary components (X-Y) or in a ternary plot.

    Todo:
        Split logscales for x and y (currently only for log-log)

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe from which to draw data.
    components: list, None
        Elements or compositional components to plot.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    mode: str, 'density'
        Different modes used here: ['density', 'hexbin', 'hist2d']
    coverage_scale: float, 1.2
        Scale the area over which the density plot is drawn.
    """
    kwargs = kwargs.copy()
    df = to_frame(df)
    try:
        if df.columns.size not in [2, 3]:
            assert len(components) in [2, 3]

        if components is None:
            components = df.columns.values
    except:
        raise AssertionError('Please suggest elements or provide a slice of the dataframe.')

    figsize = kwargs.pop('figsize', 8.)
    fontsize = kwargs.pop('fontsize', 12.)
    linewidths = kwargs.pop('linewidths', None)
    linestyles = kwargs.pop('linestyles', None)

    colorbar = kwargs.pop('colorbar', False)

    ax = ax or plt.subplots(1, figsize=(figsize, figsize* 3**0.5 * 0.5))[1]
    background_color = ax.patch.get_facecolor()
    nbins = kwargs.pop('bins', 20)
    cmap = kwargs.pop('cmap', DEFAULT_CONT_COLORMAP)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap.set_under(color=background_color)

    exp = (coverage_scale-1)/2
    data = df.loc[:, components].values

    if data.any():
        # Data can't be plotted if there's any nans, so we can exclude these
        data = data[(np.isfinite(data).all(axis=1)), :]
        if len(components) == 2:  # binary
            x, y = data.T
            xmin, xmax = on_finite(x, np.min)*(1.-exp), \
                         on_finite(x, np.max)*(1.+exp) # 120% range
            ymin, ymax = on_finite(y, np.min)*(1.-exp), \
                         on_finite(y, np.max)*(1.+exp) # 120% range
            xstep = (xmax-xmin) / nbins
            ystep = (ymax-ymin) / nbins

            extent = (xmin, xmax, ymin, ymax)

            if mode == 'hexbin':
                vmin = kwargs.pop('vmin', 0)
                if logspace:
                    # extent values are exponents (i.e. 3 -> 10**3)
                    hex_extent = (np.log(xmin - xstep), np.log(xmax + xstep),
                                  np.log(ymin - ystep), np.log(ymax + ystep))
                    mappable = ax.hexbin(x, y,
                                         gridsize=nbins,
                                         cmap=cmap,
                                         extent=hex_extent,
                                         xscale='log',
                                         yscale='log',
                                         **kwargs)
                else:
                    hex_extent = (xmin - xstep, xmax + xstep,
                                  ymin - ystep, ymax + ystep)
                    mappable = ax.hexbin(x, y,
                                         gridsize=nbins,
                                         cmap=cmap,
                                         extent=hex_extent,
                                         **kwargs)
                cbarlabel = 'Frequency'

            elif mode == 'hist2d':
                vmin = kwargs.pop('vmin', 0)
                if logspace:
                    assert (xmin-xstep > 0.) and (ymin-ystep > 0.)
                    xe = np.logspace(np.log(xmin-xstep),
                                     np.log(xmax+xstep),
                                     nbins+1)
                    ye = np.logspace(np.log(ymin-ystep),
                                     np.log(ymax+ystep),
                                     nbins+1)
                else:
                    xe = np.linspace(xmin-xstep, xmax+xstep, nbins+1)
                    ye = np.linspace(ymin-ystep, ymax+ystep, nbins+1)
                h, xe, ye, im = ax.hist2d(x, y,
                                          bins=[xe, ye],
                                          cmap=cmap,
                                          **kwargs)
                mappable = im
                cbarlabel = 'Frequency'

            elif mode == 'density':
                shading = kwargs.pop('shading', None) or 'flat'
                kdedata = data.T
                # Can't have nans or infs
                #kdedata = kdedata[:, (np.isfinite(kdedata).all(axis=0))]

                if logspace:
                    assert xmin > 0. and ymin > 0.
                    # Generate grid in logspace
                    _xs = np.linspace(np.log(xmin), np.log(xmax), nbins+1)
                    _ys = np.linspace(np.log(ymin), np.log(ymax), nbins+1)
                    xi, yi = np.meshgrid(_xs, _ys)
                    # Generate KDE in logspace
                    k = gaussian_kde(np.log(kdedata))
                    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                    # Revert coordinates bacl to non-log space
                    xi, yi = np.exp(xi), np.exp(yi)
                    assert np.isfinite(xi).all() and np.isfinite(yi).all()
                else:
                    xi, yi = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
                    k = gaussian_kde(kdedata)
                    zi = k(np.vstack([xi.flatten(), yi.flatten()]))


                vmin = kwargs.pop('vmin', 0.02 * np.nanmax(zi)) # 1% max height
                if contour:
                    levels = kwargs.pop('levels', None)

                    if levels is None:
                        levels = MaxNLocator(nbins=10).tick_values(zi.min(),
                                                                   zi.max())
                    elif isinstance(levels, int):
                        levels = MaxNLocator(nbins=levels).tick_values(zi.min(),
                                                                       zi.max())

                    mappable = ax.contourf(xi, yi, zi.reshape(xi.shape),
                                           extent=extent,
                                           levels=levels,
                                           vmin=vmin,
                                           **kwargs)

                    ax.contour(xi, yi, zi.reshape(xi.shape),
                               extent=extent,
                               linewidths=linewidths,
                               linestyles=linestyles,
                               vmin=vmin,
                               **kwargs)
                else:
                    # updated to pcolor to avoid quadmesh issues with alpha
                    mappable = ax.pcolormesh(xi, yi, zi.reshape(xi.shape),
                                             cmap=cmap,
                                             shading=shading,
                                             vmin=vmin,
                                             **kwargs)
                    mappable.set_edgecolor(background_color)
                    mappable.set_linestyle('None')
                    mappable.set_lw(0.)


                cbarlabel = 'Kernel Density Estimate'

            if colorbar:
                add_colorbar(mappable, label=cbarlabel, **kwargs)

            ax.axis(extent)

            ax.set_xlabel(components[0], fontsize=fontsize)
            ax.set_ylabel(components[1], fontsize=fontsize)

        elif len(components) == 3:  # ternary
            scale = kwargs.pop('scale', None) or 100.
            empty_df = pd.DataFrame(columns=df.columns)
            heatmapdata = tern_heatmapcoords(data.T, scale=nbins, bins=nbins)
            tax = ternaryplot(empty_df, ax=ax, components=components, scale=scale)
            ax = tax.ax
            if mode == 'hexbin':
                style = 'hexagonal'
            else:
                style = 'triangular'
            tax.heatmap(heatmapdata, scale=scale, style=style, colorbar=colorbar, **kwargs)
        else:
            pass
    plt.tight_layout()
    return ax
