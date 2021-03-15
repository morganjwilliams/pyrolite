"""
Functions for dealing with kernel density estimation.

Attributes
----------
USE_PCOLOR : :class:`bool`
    Option to use the :func:`matplotlib.pyplot.pcolor` function in place
    of :func:`matplotlib.pyplot.pcolormesh`.
"""
import numpy as np
from numpy.linalg import LinAlgError
import scipy.interpolate
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from ..meta import subkwargs
from ..math import flattengrid, linspc_, logspc_, interpolate_line
from ..distributions import sample_kde
from .grid import bin_centres_to_edges
from ..log import Handle

logger = Handle(__name__)

try:
    import statsmodels.api as sm

    HAVE_SM = True
except ImportError:
    HAVE_SM = False

USE_PCOLOR = False


def get_axis_density_methods(ax):
    """
    Get the relevant density and contouring methods for a given axis.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes` | :class:`mpltern.ternary.TernaryAxes`
        Axis to check.

    Returns
    --------
    pcolor, contour, contourf
        Relevant functions for this axis.
    """
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


def percentile_contour_values_from_meshz(
    z, percentiles=[0.95, 0.66, 0.33], resolution=1000
):
    """
    Integrate a probability density distribution Z(X,Y) to obtain contours in Z which
    correspond to specified percentile contours. Contour values will be returned
    with the same order as the inputs.

    Parameters
    ----------
    z : :class:`numpy.ndarray`
        Probability density function over x, y.
    percentiles : :class:`numpy.ndarray`
        Percentile values for which to create contours.
    resolution : :class:`int`
        Number of bins for thresholds between 0. and max(Z)

    Returns
    -------
    labels : :class:`list`
        Labels for contours (percentiles, if above minimum z value).
    contours : :class:`list`
        Contour height values.

    Todo
    -----
    This may error for a list of percentiles where one or more requested
    values are below the miniumum threshold. The exception handling should
    be updated to cater for arrays - where some of the values may be above
    the minimum.
    """
    # Integral approach from https://stackoverflow.com/a/37932566
    t = np.linspace(0.0, z.max(), resolution)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
    f = scipy.interpolate.interp1d(integral, t)
    try:
        t_contours = f(np.array(percentiles) * z.sum())
        return percentiles, t_contours
    except ValueError:
        # occurrs on the low-end of percentiles (high parts of distribution)
        # maximum positions of distributions are limited by the resolution
        # at some point there's a step down to zero
        logger.debug(
            "Percentile contour below minimum for given resolution"
            "Returning Minimium."
        )
        non_one = integral[~np.isclose(integral, np.ones_like(integral))]
        return ["min"], f(np.array([np.nanmax(non_one)]))


def plot_Z_percentiles(
    *coords,
    zi=None,
    percentiles=[0.95, 0.66, 0.33],
    ax=None,
    extent=None,
    fontsize=8,
    cmap=None,
    colors=None,
    linewidths=None,
    linestyles=None,
    contour_labels=None,
    label_contours=True,
    **kwargs
):
    """
    Plot percentile contours onto a 2D  (scaled or unscaled) probability density
    distribution Z over X,Y.

    Parameters
    ------------
    coords : :class:`numpy.ndarray`
        Arrays of (x, y) or (a, b, c) coordinates.
    z : :class:`numpy.ndarray`
        Probability density function over x, y.
    percentiles : :class:`list`
        Percentile values for which to create contours.
    ax : :class:`matplotlib.axes.Axes`, :code:`None`
        Axes on which to plot. If none given, will create a new Axes instance.
    extent : :class:`list`, :code:`None`
        List or np.ndarray in the form [-x, +x, -y, +y] over which the image extends.
    fontsize : :class:`float`
        Fontsize for the contour labels.
    cmap : :class:`matplotlib.colors.ListedColormap`
        Color map for the contours and contour labels.
    colors : :class:`str` | :class:`list`
        Colors for the contours, can optionally be specified *in place of* `cmap.`
    linewidths : :class:`str` | :class:`list`
        Widths of contour lines.
    linestyles : :class:`str` | :class:`list`
        Styles for contour lines.
    contour_labels : :class:`dict` | :class:`list`
        Labels to assign to contours, organised by level.
    label_contours :class:`bool`
        Whether to add text labels to individual contours.

    Returns
    -------
    :class:`matplotlib.contour.QuadContourSet`
        Plotted and formatted contour set.

    Notes
    -----
    When the contours are percentile based, high percentile contours tend to get
    washed our with colormapping - consider adding different controls on coloring,
    especially where there are only one or two contours specified. One way to do
    this would be via the string based keyword argument `colors` for plt.contour, with
    an adaption for non-string colours which post-hoc modifies the contour lines
    based on the specified colours?
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 6))

    if extent is None:
        # if len(coords) == 2:  # currently won't work for ternary
        extent = np.array([[np.min(c), np.max(c)] for c in coords[:2]]).flatten()

    clabels, contour_values = percentile_contour_values_from_meshz(
        zi, percentiles=percentiles
    )

    pcolor, contour, contourf = get_axis_density_methods(ax)
    if colors is not None:  # colors are explicitly specified
        cmap = None

    # contours will need to increase for matplotlib, so we check the ordering here.
    ordering = np.argsort(contour_values)
    # sort out multi-object properties - reorder to fit the increasing order requirement
    cntr_config = {}
    for p, v in [
        ("colors", colors),
        ("linestyles", linestyles),
        ("linewidths", linewidths),
    ]:
        if v is not None:
            if isinstance(v, (list, tuple)):
                # reorder the list
                cntr_config[p] = [v[ix] for ix in ordering]
            else:
                cntr_config[p] = v

    cs = contour(
        *coords,
        zi,
        levels=contour_values[ordering],  # must increase
        cmap=cmap,
        **{**cntr_config, **kwargs}
    )
    if label_contours:
        fs = kwargs.pop("fontsize", None) or 8
        lbls = ax.clabel(cs, fontsize=fs, inline_spacing=0)
        z_contours = sorted(list(set([float(l.get_text()) for l in lbls])))
        trans = {
            float(t): str(p)
            for t, p in zip(z_contours, sorted(percentiles, reverse=True))
        }
        if contour_labels is None:
            _labels = [trans[float(l.get_text())] for l in lbls]
        else:
            if isinstance(contour_labels, dict):
                # get the labels from the dictionary provided
                contour_labels = {str(k): str(v) for k, v in contour_labels.items()}
                _labels = [contour_labels[trans[float(l.get_text())]] for l in lbls]
            else:  # a list is specified in the same order as the contours are drawn
                _labels = contour_labels

        for l, t in zip(lbls, _labels):
            l.set_text(t)
    return cs


def conditional_prob_density(
    y,
    x=None,
    logy=False,
    resolution=5,
    bins=50,
    yextent=None,
    rescale=True,
    mode="binkde",
    ret_centres=False,
    **kwargs
):
    """
    Estimate the conditional probability density of one dependent variable.

    Parameters
    -----------
    y : :class:`numpy.ndarray`
        Dependent variable for which to calculate conditional probability P(y | X=x)
    x : :class:`numpy.ndarray`, :code:`None`
        Optionally-specified independent index.
    logy : :class:`bool`
        Whether to use a logarithmic bin spacing on the y axis.
    resolution : :class:`int`
        Points added per segment via interpolation along the x axis.
    bins : :class:`int`
        Bins for histograms and grids along the independent axis.
    yextent : :class:`tuple`
        Extent in the y direction.
    rescale : :class:`bool`
        Whether to rescale bins to give the same max Z across x.
    mode : :class:`str`
        Mode of computation.

            If mode is :code:`"ckde"`, use
            :func:`statsmodels.nonparametric.KDEMultivariateConditional` to compute a
            conditional kernel density estimate. If mode is :code:`"kde"`, use a normal
            gaussian kernel density estimate. If mode is :code:`"binkde"`, use a gaussian
            kernel density estimate over y for each bin. If mode is :code:`"hist"`,
            compute a histogram.
    ret_centres : :class:`bool`
        Whether to return bin centres in addtion to histogram edges,
        e.g. for later contouring.

    Returns
    -------
    :class:`tuple` of :class:`numpy.ndarray`
        :code:`x` bin edges :code:`xe`, :code:`y` bin edges :code:`ye`, histogram/density
        estimates :code:`Z`. If :code:`ret_centres` is :code:`True`, the last two return
        values will contain the bin centres :code:`xi`, :code:`yi`.
    """
    # check for shapes
    assert not ((x is None) and (y is None))
    if y is None:  # Swap the variables. Create an index for x
        y = x
        x = None

    nvar = y.shape[1]
    if x is None:  # Create a simple arange-based index
        x = np.arange(nvar)

    if resolution:  # this is where REE previously broke down
        x, y = interpolate_line(x, y, n=resolution, logy=logy)

    if not x.shape == y.shape:
        try:  # x is an index to be tiled
            assert y.shape[1] == x.shape[0]
            x = np.tile(x, y.shape[0]).reshape(*y.shape)
        except AssertionError:
            # shape mismatch
            msg = "Mismatched shapes: x: {}, y: {}. Needs either ".format(
                x.shape, y.shape
            )
            raise AssertionError(msg)

    xx = x[0]
    if yextent is None:
        ymin, ymax = np.nanmin(y), np.nanmax(y)
    else:
        ymin, ymax = np.nanmin(yextent), np.nanmax(yextent)

    # remove non finite values for kde functions
    ystep = [(ymax - ymin) / bins, (ymax / ymin) / bins][logy]
    yy = [linspc_, logspc_][logy](ymin, ymax, step=ystep, bins=bins)
    if logy:  # make grid equally spaced, evaluate in log then transform back
        y, yy = np.log(y), np.log(yy)

    xi, yi = np.meshgrid(xx, yy)
    # bin centres may be off centre, but will be in the bins.
    xe, ye = np.meshgrid(bin_centres_to_edges(xx, sort=False), bin_centres_to_edges(yy))

    kde_kw = subkwargs(kwargs, sample_kde)

    if mode == "ckde":
        fltr = np.isfinite(y.flatten()) & np.isfinite(x.flatten())
        x, y = x.flatten()[fltr], y.flatten()[fltr]
        if HAVE_SM:
            dens_c = sm.nonparametric.KDEMultivariateConditional(
                endog=[y], exog=[x], dep_type="c", indep_type="c", bw="normal_reference"
            )
        else:
            raise ImportError("Requires statsmodels.")
        # statsmodels pdf takes values in reverse order
        zi = dens_c.pdf(yi.flatten(), xi.flatten()).reshape(xi.shape)
    elif mode == "binkde":  # calclate a kde per bin
        zi = np.zeros(xi.shape)
        for bin_index in range(x.shape[1]):  # bins along the x-axis
            # if np.isfinite(y[:, bin_index]).any(): # bins can be empty
            src = y[:, bin_index]
            sample_at = yi[:, bin_index]
            zi[:, bin_index] = sample_kde(src, sample_at, **kde_kw)
            # else:
            # pass
    elif mode == "kde":  # eqivalent to 2D KDE for scatter x,y * resolution
        xkde = sample_kde(x[0], x[0])  # marginal density along x
        src = np.vstack([x.flatten(), y.flatten()]).T
        sample_at = [xi, yi]  # meshgrid logistics dealt with by sample_kde
        try:
            zi = sample_kde(src, sample_at, **kde_kw)
        except LinAlgError:  # singular matrix, try adding miniscule noise on x?
            logger.warn("Singular Matrix")
            src[:, 0] += np.random.randn(*x.shape) * np.finfo(np.float).eps
        zi = sample_kde(src, sample_at, **kde_kw)
        zi.reshape(xi.shape)
        zi /= xkde[np.newaxis, :]
    elif "hist" in mode.lower():  # simply compute the histogram
        # histogram monotonically increasing bins, requires logbins be transformed
        # calculate histogram in logy if needed
        bins = [bin_centres_to_edges(xx), bin_centres_to_edges(yy)]
        H, xe, ye = np.histogram2d(x.flatten(), y.flatten(), bins=bins)
        zi = H.T.reshape(xi.shape)
    else:
        raise NotImplementedError

    if rescale:  # rescale bins across x
        xzfactors = np.nanmax(zi) / np.nanmax(zi, axis=0)
        zi *= xzfactors[np.newaxis, :]

    if logy:
        yi, ye = np.exp(yi), np.exp(ye)
    if ret_centres:
        return xe, ye, zi, xi, yi
    return xe, ye, zi
