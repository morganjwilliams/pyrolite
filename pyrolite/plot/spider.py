import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
import matplotlib.collections
import numpy as np
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

from ..geochem.ind import get_ionic_radii, REE
from ..util.types import iscollection
from ..util.plot.style import DEFAULT_CONT_COLORMAP, _mpl_sp_kw_split, patchkwargs
from ..util.plot.density import (
    conditional_prob_density,
    plot_Z_percentiles,
    percentile_contour_values_from_meshz,
)
from ..util.plot.axes import get_twins, init_axes

from ..util.meta import get_additional_params, subkwargs


def spider(
    arr,
    indexes=None,
    ax=None,
    color=None,
    cmap=DEFAULT_CONT_COLORMAP,
    norm=None,
    alpha=None,
    marker="D",
    markersize=5.0,
    label=None,
    logy=True,
    yextent=None,
    mode="plot",
    unity_line=False,
    **kwargs
):
    """
    Plots spidergrams for trace elements data. Additional arguments are typically forwarded
    to respective :mod:`matplotlib` functions :func:`~matplotlib.pyplot.plot` and
    :func:`~matplotlib.pyplot.scatter` (see Other Parameters, below).

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        Data array.
    indexes : : :class:`numpy.ndarray`
        Numerical indexes of x-axis positions.
    ax : :class:`matplotlib.axes.Axes`, :code:`None`
        The subplot to draw on.
    color : :class:`str` | :class:`list` | :class:`numpy.ndarray`
        Individual color or collection of :mod:`~matplotlib.colors` to be passed to matplotlib.
    cmap : :class:`matplotlib.colors.Colormap`
        Colormap for mapping point and line colors.
    norm : :class:`matplotlib.colors.Normalize`, :code:`None`
        Normalization instane for the colormap.
    marker : :class:`str`, 'D'
        Matplotlib :mod:`~matplotlib.markers` designation.
    markersize : :class:`int`, 5.
        Size of individual markers.
    label : :class:`str`, :code:`None`
        Label for the individual series.
    logy : :class:`bool`
        Whether to use a log y-axis.
    yextent : :class:`tuple`
        Extent in the y direction for conditional probability plots.
    mode : :class:`str`,  :code:`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
        Mode for plot. Plot will produce a line-scatter diagram. Fill will return
        a filled range. Density will return a conditional density diagram.
    unity_line : :class:`bool`
        Add a line at y=1 for reference.

    {otherparams}

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the spiderplot is plotted.

    Notes
    -----
        By using separate lines and scatterplots, values between two missing
        items are still presented.

    Todo
    ----
        * Might be able to speed up lines with `~matplotlib.collections.LineCollection`.
        * Legend entries

    .. seealso::

        Functions:

            :func:`matplotlib.pyplot.plot`
            :func:`matplotlib.pyplot.scatter`
            :func:`REE_v_radii`
    """

    # ---------------------------------------------------------------------
    ncomponents = arr.shape[-1]
    figsize = kwargs.pop("figsize", None) or (ncomponents * 0.3, 4)

    ax = init_axes(ax=ax, figsize=figsize, **kwargs)

    if logy:
        ax.set_yscale("log")

    if indexes is None:
        indexes = np.arange(ncomponents)
    else:
        indexes = np.array(indexes)

    if indexes.ndim == 1:
        indexes0 = indexes
    else:
        indexes0 = indexes[0]

    ax.set_xticks(indexes0)

    # if there is no data, return the blank axis
    if (arr is None) or (not np.isfinite(arr).sum()):
        return ax

    if indexes.ndim < arr.ndim:
        indexes = np.tile(indexes0, (arr.shape[0], 1))

    local_kw = dict(  # register aliases
        c=color, color=color, marker=marker, markersize=markersize, s=markersize ** 2
    )
    local_kw = {**local_kw, **kwargs}
    if local_kw.get("color") is None and local_kw.get("c") is None:
        local_kw["color"] = next(ax._get_lines.prop_cycler)["color"]

    sctkw, lnkw = _mpl_sp_kw_split(local_kw)
    _ = lnkw.pop("label")
    # check if colors vary per line/sctr
    variable_colors = False
    c = sctkw.get("c", None)

    if c is not None:
        if iscollection(c):
            variable_colors = True

    if unity_line:
        ax.axhline(1.0, ls="--", c="k", lw=0.5)

    if "fill" in mode.lower():
        mins = np.nanmin(arr, axis=0)
        maxs = np.nanmax(arr, axis=0)
        plycol = ax.fill_between(indexes0, mins, maxs, **patchkwargs(local_kw))
    elif "plot" in mode.lower():
        ls = ax.plot(indexes.T, arr.T, **lnkw)
        if variable_colors:
            # perhaps check shape of color arg here
            for l, ic in zip(ls, c):
                l.set_color(ic)

        sctkw.update(dict(label=label))
        sc = ax.scatter(indexes.T, arr.T, **sctkw)
        # should create a custom legend handle here

        # could modify legend here.
    elif any([i in mode.lower() for i in ["binkde", "ckde", "kde", "hist"]]):
        if "contours" in kwargs and "vmin" in kwargs:
            msg = "Combining `contours` and `vmin` arugments for density plots should be avoided."
            logger.warn(msg)
        xe, ye, zi, xi, yi = conditional_prob_density(
            arr,
            x=indexes0,
            logy=logy,
            yextent=yextent,
            mode=mode,
            ret_centres=True,
            **local_kw
        )
        # can have issues with nans here?
        vmin = kwargs.pop("vmin", 0)
        vmin = percentile_contour_values_from_meshz(zi, [1.0 - vmin])[1][0]  # pctl
        if "contours" in kwargs:
            pzpkwargs = {  # keyword arguments to forward to plot_Z_percentiles
                **subkwargs(kwargs, plot_Z_percentiles),
                **{"percentiles": kwargs["contours"]},
            }
            plot_Z_percentiles(  # pass all relevant kwargs including contours
                xi, yi, zi=zi, ax=ax, cmap=cmap, vmin=vmin, **pzpkwargs
            )
        else:
            zi[zi < vmin] = np.nan
            ax.pcolormesh(
                xe, ye, zi, cmap=cmap, vmin=vmin, *subkwargs(kwargs, ax.pcolormesh)
            )
    else:
        raise NotImplementedError(
            "Accepted modes: {plot, fill, binkde, ckde, kde, hist}"
        )

    # consider relimiting here
    return ax


def REE_v_radii(
    arr=None,
    ax=None,
    ree=REE(),
    index="elements",
    mode="plot",
    logy=True,
    tl_rotation=60,
    unity_line=False,
    **kwargs
):
    r"""
    Creates an axis for a REE diagram with ionic radii along the x axis.

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        Data array.
    ax : :class:`matplotlib.axes.Axes`, :code:`None`
        Optional designation of axes to reconfigure.
    ree : :class:`list`
        List of REE to use as an index.
    index : :class:`str`
        Whether to plot using radii on the x-axis ('radii'), or elements ('elements').
    mode : :class:`str`, :code:`["plot", "fill", "binkde", "ckde", "kde", "hist"]`
        Mode for plot. Plot will produce a line-scatter diagram. Fill will return
        a filled range. Density will return a conditional density diagram.
    logy : :class:`bool`
        Whether to use a log y-axis.
    tl_rotation : :class:`float`
        Rotation of the numerical index labels in degrees.
    unity_line : :class:`bool`
        Add a line at y=1 for reference.

    {otherparams}

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the REE_v_radii plot is added.

    Todo
    ----
        * Turn this into a plot template within pyrolite.plot.templates submodule

    .. seealso::

        Functions:

            :func:`matplotlib.pyplot.plot`
            :func:`matplotlib.pyplot.scatter`
            :func:`spider`
            :func:`pyrolite.geochem.transform.lambda_lnREE`

    """
    ax = init_axes(ax=ax, **kwargs)

    radii = np.array(get_ionic_radii(ree, charge=3, coordination=8))

    xlabels, _xlabels = ["{:1.3f}".format(i) for i in radii], ree
    xticks, _xticks = radii, radii
    xlim = (0.99 * np.min(radii), 1.01 * np.max(radii))
    xlabelrotation, _xlabelrotation = tl_rotation, 0
    xtitle, _xtitle = r"Ionic Radius ($\mathrm{\AA}$)", "Element"

    if index == "radii":
        invertx = False
        indexes = radii
    else:  # mode == 'elements'
        invertx = True
        indexes = radii
        # swap ticks labels etc,
        _xtitle, xtitle = xtitle, _xtitle
        _xlabels, xlabels = xlabels, _xlabels
        _xlabelrotation, xlabelrotation = xlabelrotation, _xlabelrotation

    if arr is not None:
        kwargs["indexes"] = kwargs.get("indexes", indexes)
        ax = spider(arr, ax=ax, logy=logy, mode=mode, unity_line=unity_line, **kwargs)

    ax.set_xlabel(xtitle)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=xlabelrotation)
    if invertx:
        xlim = xlim[::-1]
    if xlim is not None:
        ax.set_xlim(xlim)

    twinys = get_twins(ax, which="y")
    if len(twinys):
        _ax = twinys[0]
    else:
        _ax = ax.twiny()
    _ax.set_xlabel(_xtitle)
    _ax.set_xticks(_xticks)
    _ax.set_xticklabels(_xlabels, rotation=_xlabelrotation)
    _ax.set_xlim(ax.get_xlim())
    return ax


_add_additional_parameters = True
spider.__doc__ = spider.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            spider,
            plt.scatter,
            plt.plot,
            matplotlib.lines.Line2D,
            indent=4,
            header="Other Parameters",
            subsections=True,
        ),
    ][_add_additional_parameters]
)

REE_v_radii.__doc__ = REE_v_radii.__doc__.format(
    otherparams=[
        "",
        get_additional_params(
            REE_v_radii, spider, indent=4, header="Other Parameters", subsections=True
        ),
    ][_add_additional_parameters]
)
