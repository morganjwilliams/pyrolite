import matplotlib.pyplot as plt
import numpy as np
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

from ..geochem import get_ionic_radii, REE
from ..util.general import iscollection
from ..util.plot import __DEFAULT_CONT_COLORMAP__, __DEFAULT_DISC_COLORMAP__
from ..util.meta import get_additional_params


def spider(
    arr,
    indexes=None,
    ax=None,
    color="0.5",
    cmap=__DEFAULT_CONT_COLORMAP__,
    norm=None,
    alpha=1.0,
    marker="D",
    markersize=5.0,
    label=None,
    figsize=None,
    logy=True,
    plot=True,
    fill=False,
    density=False,
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
    ax : :class:`matplotlib.axes.Axes`, `None`
        The subplot to draw on.
    color : :class:`str` | :class:`list` | :class:`numpy.ndarray`
        Individual color or collection of :mod:`~matplotlib.colors` to be passed to matplotlib.
    cmap : :class:`matplotlib.colors.Colormap`
        Colormap for mapping point and line colors.
    norm : :class:`matplotlib.colors.Normalize`, `None`
        Normalization instane for the colormap.
    alpha : :class:`float`, 1.
        Opacity for the plotted series.
    marker : :class:`str`, 'D'
        Matplotlib :mod:`~matplotlib.markers` designation.
    markersize : :class:`int`, 5.
        Size of individual markers.
    label : :class:`str`, `None`
        Label for the individual series.
    figsize : :class:`tuple`, `None`
        Size of the figure to be generated, if not using an existing :class:`~matplotlib.axes.Axes`.
    plot : :class:`bool`, `True`
        Whether to plot lines and markers.
    fill : :class:`bool`, `False`
        Whether to add a patch representing the full range.
    density : :class:`bool`, `False`
        Whether to add a conditional kernel density estimate over the index.

    {otherparams}

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the spiderplot is plotted.

    Note
    -----
        By using separate lines and scatterplots, values between two missing
        items are still presented.

    Todo
    ----
        * Might be able to speed up lines with `~matplotlib.collections.LineCollection`.
        * Conditional density plot.

    See Also
    ---------
    :func:`matplotlib.pyplot.plot`
    :func:`matplotlib.pyplot.scatter`
    :func:`REE_radii_plot`
    """

    # ---------------------------------------------------------------------
    ncomponents = arr.shape[-1]
    figsize = figsize or (ncomponents * 0.3, 4)

    ax = ax or plt.subplots(1, figsize=figsize)[1]

    if logy:
        ax.set_yscale("log")

    if indexes is None:
        indexes = np.arange(ncomponents)

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

    try:
        assert plot or fill
    except:
        msg = "Please select to either plot values or fill between ranges."
        raise AssertionError(msg)

    sty = {}

    # Color ----------------------------------------------------------

    variable_colors = False
    if color is not None:
        if iscollection(color):
            sty["c"] = color
            variable_colors = True
        else:
            sty["color"] = color

    _c = sty.pop("c", None)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if (_c is not None) and (cmap is not None):
        if norm is not None:
            _c = [norm(c) for c in _c]
        _c = [cmap(c) for c in _c]

    sty["alpha"] = alpha

    if fill:
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        plycol = ax.fill_between(indexes0, mins, maxs, **sty)
        # Use the first (typically only) element for color
        if (sty.get("color") is None) and (sty.get("c") is None):
            sty["color"] = plycol.get_facecolor()[0]

    sty["marker"] = marker

    # Use the default color cycling to provide a single color
    if sty.get("color") is None and _c is None:
        sty["color"] = next(ax._get_lines.prop_cycler)["color"]

    if plot:
        ls = ax.plot(indexes0, arr.T, **sty)
        if variable_colors:
            for l, c in zip(ls, _c):
                l.set_color(c)

        sty["s"] = markersize
        if (sty.get("color") is None) and (_c is None):
            sty["color"] = ls[0].get_color()

        sty["label"] = label
        # For the scatter, the number of points > the number of series
        # Need to check if this is the case, and create equivalent

        if _c is not None:
            cshape = np.array(_c).shape
            if cshape != df.loc[:, components].shape:
                # expand it across the columns
                _c = np.tile(_c, (len(components), 1))

        sc = ax.scatter(indexes.T, arr.T, **sty)

    unused_keys = [i for i in kwargs if i not in list(sty.keys())]
    if len(unused_keys):
        logger.info("Styling not yet implemented for:{}".format(unused_keys))

    return ax


def REE_v_radii(
    arr=None, ax=None, ree=REE(), mode="radii", tl_rotation=60, **kwargs
):
    """
    Creates an axis for a REE diagram with ionic radii along the x axis.

    Parameters
    -----------
    arr : :class:`np.ndarray`
        Data array.
    ax : :class:`matplotlib.axes.Axes`
        Optional designation of axes to reconfigure.
    ree : :class:`list`
        List of REE to use as an index.
    mode : :class:`str`
        Whether to plot using radii on the x-axis ('radii'), or elements ('elements').
    tl_rotation : :class:`float`
        Rotation of the numerical index labels in degrees.

    {otherparams}

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes on which the REE_v_radii plot is added.

    See Also
    ---------
    :func:`matplotlib.pyplot.plot`
    :func:`matplotlib.pyplot.scatter`
    :func:`spider`
    :func:`pyrolite.geochem.transform.lambda_lnREE`
    """
    if ax is not None:
        fig = ax.figure
        ax = ax
    else:
        fig, ax = plt.subplots()

    radii = np.array(get_ionic_radii(ree, charge=3, coordination=8))

    xlabels, _xlabels = ["{:1.3f}".format(i) for i in radii], ree
    xticks, _xticks = radii, radii
    xlabelrotation, _xlabelrotation = tl_rotation, 0
    xtitle, _xtitle = "Ionic Radius ($\mathrm{\AA}$)", "Element"

    if mode == "radii":
        indexes = radii
        xlim = (0.99 * np.min(radii), 1.01 * np.max(radii))
    else:  # mode == 'elements'

        indexes = None
        xlim = None
        # swap ticks labels etc,
        _xtitle, xtitle = xtitle, _xtitle
        _xlabels, xlabels = xlabels, _xlabels
        _xticks, xticks = np.arange(len(ree)), np.arange(len(ree))
        _xlabelrotation, xlabelrotation = xlabelrotation, _xlabelrotation

    if arr is not None:
        ax = spider(arr, indexes=indexes, ax=ax, logy=True, **kwargs)

    ax.axhline(1.0, ls="--", c="k", lw=0.5)
    ax.set_xlabel(xtitle)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=xlabelrotation)
    if xlim is not None:
        ax.set_xlim(xlim)
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
            spider, plt.scatter, indent=4, header="Other Parameters", subsections=True
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
