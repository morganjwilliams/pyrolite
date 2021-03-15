import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
import matplotlib.collections
import numpy as np
from ..util.log import Handle

logger = Handle(__name__)


from .color import process_color
from ..geochem.ind import get_ionic_radii, REE
from ..util.types import iscollection
from ..util.plot.style import (
    DEFAULT_CONT_COLORMAP,
    linekwargs,
    scatterkwargs,
    patchkwargs,
)
from ..util.plot.density import (
    conditional_prob_density,
    plot_Z_percentiles,
    percentile_contour_values_from_meshz,
)
from ..util.plot.axes import get_twins, init_axes

from ..util.meta import get_additional_params, subkwargs

_scatter_defaults = dict(cmap=DEFAULT_CONT_COLORMAP, marker="D", s=25,)
_line_defaults = dict(cmap=DEFAULT_CONT_COLORMAP)

# could create a spidercollection?
def spider(
    arr,
    indexes=None,
    ax=None,
    label=None,
    logy=True,
    yextent=None,
    mode="plot",
    unity_line=False,
    scatter_kw={},
    line_kw={},
    set_ticks=True,
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
    scatter_kw : :class:`dict`
        Keyword parameters to be passed to the scatter plotting function.
    line_kw : :class:`dict`
        Keyword parameters to be passed to the line plotting function.
    set_ticks : :class:`bool`
        Whether to set the x-axis ticks according to the specified index.
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

    if unity_line:
        ax.axhline(1.0, ls="--", c="k", lw=0.5)

    if indexes is None:
        indexes = np.arange(ncomponents)
    else:
        indexes = np.array(indexes)

    if indexes.ndim == 1:
        indexes0 = indexes
    else:
        indexes0 = indexes[0]

    if set_ticks:
        ax.set_xticks(indexes0)

    # if there is no data, return the blank axis
    if (arr is None) or (not np.isfinite(arr).sum()):
        return ax

    # if the indexes are supplied as a 1D array but the data is 2D, we need to expand
    # it to fit the scatter data
    if indexes.ndim < arr.ndim:
        indexes = np.tile(indexes0, (arr.shape[0], 1))

    if "fill" in mode.lower():
        mins = np.nanmin(arr, axis=0)
        maxs = np.nanmax(arr, axis=0)
        plycol = ax.fill_between(indexes0, mins, maxs, **patchkwargs(kwargs))
    elif "plot" in mode.lower():
        # copy params
        l_kw, s_kw = {**line_kw}, {**scatter_kw}
        ################################################################################
        if line_kw.get("cmap") is None:
            l_kw["cmap"] = kwargs.get("cmap", None)

        l_kw = {**kwargs, **l_kw}

        # if a line color hasn't been specified, perhaps we can use the scatter 'c'
        if l_kw.get("color") is None:
            if l_kw.get("c") is not None:
                l_kw["color"] = kwargs.get("c")
        if "c" in l_kw:
            l_kw.pop("c")  # remove c if it's been specified globally
        # if a color option is not specified, get the next cycled color
        if l_kw.get("color") is None:
            # add cycler color as array to suppress singular color warning
            l_kw["color"] = np.array([next(ax._get_lines.prop_cycler)["color"]])

        l_kw = linekwargs(process_color(**{**_line_defaults, **l_kw}))
        # marker explictly dealt with by scatter
        for k in ["marker", "markers"]:
            l_kw.pop(k, None)
        # Construct and Add LineCollection?
        lcoll = matplotlib.collections.LineCollection(
            np.dstack((indexes, arr)), **{"zorder": 1, **l_kw}
        )
        ax.add_collection(lcoll)

        ################################################################################
        # load defaults and any specified parameters in scatter_kw / line_kw
        if s_kw.get("cmap") is None:
            s_kw["cmap"] = kwargs.get("cmap", None)

        _sctr_cfg = {**_scatter_defaults, **kwargs, **s_kw}
        s_kw = process_color(**_sctr_cfg)

        if s_kw["marker"] is not None:
            # will need to process colours for scatter markers here

            s_kw.update(dict(label=label))

            scattercolor = None
            if s_kw.get("c") is not None:
                scattercolor = s_kw.get("c")
            elif s_kw.get("color") is not None:
                scattercolor = s_kw.get("color")
            else:
                # no color recognised - will be default, here we get the
                # cycled color we added earlier
                scattercolor = l_kw["color"]

            if scattercolor is not None:
                if not isinstance(scattercolor, (str, tuple)):
                    # colors will be processed to arrays by this point
                    # here we reshape them to be the same length as ravel-ed arrays
                    if scattercolor.ndim >= 2 and scattercolor.shape[0] > 1:
                        scattercolor = np.tile(scattercolor, arr.shape[1]).reshape(
                            -1, scattercolor.shape[1]
                        )
                else:
                    # singular color should be converted to 2d array?
                    pass
            s_kw = scatterkwargs(
                {k: v for k, v in s_kw.items() if k not in ["c", "color"]}
            )
            sc = ax.scatter(
                indexes.ravel(), arr.ravel(), c=scattercolor, **{"zorder": 2, **s_kw}
            )

        # should create a custom legend handle here

        # could modify legend here.
    elif any([i in mode.lower() for i in ["binkde", "ckde", "kde", "hist"]]):
        cmap = kwargs.pop("cmap", None)
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
            **kwargs
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
                xe, ye, zi, cmap=cmap, vmin=vmin, **subkwargs(kwargs, ax.pcolormesh)
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
    scatter_kw={},
    line_kw={},
    set_labels=True,
    set_ticks=True,
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
    scatter_kw : :class:`dict`
        Keyword parameters to be passed to the scatter plotting function.
    line_kw : :class:`dict`
        Keyword parameters to be passed to the line plotting function.
    set_labels : :class:`bool`
        Whether to set the x-axis ticklabels for the REE.
    set_ticks : :class:`bool`
        Whether to set the x-axis ticks according to the specified index.
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
        ax = spider(
            arr,
            ax=ax,
            logy=logy,
            mode=mode,
            unity_line=unity_line,
            indexes=indexes,
            scatter_kw=scatter_kw,
            line_kw=line_kw,
            **kwargs
        )

    twinys = get_twins(ax, which="y")
    if len(twinys):
        _ax = twinys[0]
    else:
        _ax = ax.twiny()

    if set_labels:
        ax.set_xlabel(xtitle)
        _ax.set_xlabel(_xtitle)

    if set_ticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=xlabelrotation)

        if invertx:
            xlim = xlim[::-1]
        if xlim is not None:
            ax.set_xlim(xlim)

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
