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
from ..math import flattengrid, linspc_, logspc_, interpolate_line
from .grid import bin_centres_to_edges
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

try:
    import statsmodels.api as sm

    HAVE_SM = True
except ImportError:
    HAVE_SM = False

USE_PCOLOR = False


def percentile_contour_values_from_meshz(
    z, percentiles=[0.95, 0.66, 0.33], resolution=1000
):
    """
    Integrate a probability density distribution Z(X,Y) to obtain contours in Z which
    correspond to specified percentile contours.T

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
    """
    percentiles = sorted(percentiles, reverse=True)
    # Integral approach from https://stackoverflow.com/a/37932566
    t = np.linspace(0.0, z.max(), resolution)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
    f = scipy.interpolate.interp1d(integral, t)
    try:
        t_contours = f(np.array(percentiles) * z.sum())
        return percentiles, t_contours
    except ValueError:
        logger.debug(
            "Percentile contour below minimum for given resolution"
            "Returning Minimium."
        )
        non_one = integral[~np.isclose(integral, np.ones_like(integral))]
        return ["min"], f(np.array([np.nanmax(non_one)]))


def conditional_prob_density(
    y,
    x=None,
    logy=False,
    resolution=5,
    ybins=100,
    rescale=True,
    mode="binkde",
    ret_centres=False,
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
    ybins : :class:`int`
        Bins for histograms and grids along the independent axis.
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


    Notes
    ------

        * Bins along the x axis are defined such that the x points (including
          interpolated points) are the centres.

    Todo
    -----

        * Tests
        * Implement log grids (for y)
        * Add approach for interpolation? (need resolution etc) - this will resolve lines, not points!
    """
    # check for shapes
    assert not ((x is None) and (y is None))
    if y is None:  # Swap the variables. Create an index for x
        y = x
        x = None

    nvar = y.shape[1]
    if x is None:  # Create a simple arange-based index
        x = np.arange(nvar)

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

    if resolution:
        xy = np.array([x, y])
        xy = np.swapaxes(xy, 1, 0)
        xy = interpolate_line(xy, n=resolution, logy=logy)
        x, y = np.swapaxes(xy, 0, 1)

    xx = np.sort(x[0])
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    # remove non finite values for kde functions
    ystep = [(ymax - ymin) / ybins, (ymax / ymin) / ybins][logy]
    yy = [linspc_, logspc_][logy](ymin, ymax, step=ystep, bins=ybins)
    if logy:  # make grid equally spaced, evaluate in log then transform back
        y, yy = np.log(y), np.log(yy)
    # yy is backwards?
    xi, yi = np.meshgrid(xx, yy)
    xe, ye = np.meshgrid(bin_centres_to_edges(xx), bin_centres_to_edges(yy))

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
    elif mode == "kde":  # kde of dataset
        xkde = gaussian_kde(x[0])(x[0])  # marginal density along x
        fltr = np.isfinite(y.flatten()) & np.isfinite(x.flatten())
        x, y = x.flatten()[fltr], y.flatten()[fltr]
        try:
            kde = gaussian_kde(np.vstack([x, y]))
        except LinAlgError:  # singular matrix, need to add miniscule noise on x?
            logger.warn("Singular Matrix")
            logger.x = x + np.random.randn(*x.shape) * np.finfo(np.float).eps
            kde = gaussian_kde(np.vstack(([x, y])).T)

        zi = kde(flattengrid([xi, yi]).T).reshape(xi.shape) / xkde[np.newaxis, :]
    elif mode == "binkde":  # calclate a kde per bin
        zi = np.zeros(xi.shape)
        for bin_index in range(x.shape[1]):
            # if np.isfinite(y[:, bin_index]).any(): # bins can be empty
            kde = gaussian_kde(y[np.isfinite(y[:, bin_index]), bin_index])
            zi[:, bin_index] = kde(yi[:, bin_index])
            # else:
            # pass
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


def plot_Z_percentiles(
    *coords,
    zi=None,
    percentiles=[0.95, 0.66, 0.33],
    ax=None,
    extent=None,
    fontsize=8,
    cmap=None,
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
        Color map for the contours, contour labels and imshow.
    contour_labels : :class:`dict`
        Labels to assign to contours, organised by level.
    label_contours :class:`bool`
        Whether to add text labels to individual contours.

    Returns
    -------
    :class:`matplotlib.contour.QuadContourSet`
        Plotted and formatted contour set.

    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 6))

    if extent is None:
        # if len(coords) == 2:  # currently won't work for ternary
        extent = np.array([[np.min(c), np.max(c)] for c in coords[:2]]).flatten()

    clabels, contours = percentile_contour_values_from_meshz(
        zi, percentiles=percentiles
    )

    pcolor, contour, contourf = get_axis_density_methods(ax)
    cs = contour(*coords, zi, levels=contours, cmap=cmap, **kwargs)
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
        else:  # get the labels from the dictionary provided
            contour_labels = {str(k): str(v) for k, v in contour_labels.items()}
            _labels = [contour_labels[trans[float(l.get_text())]] for l in lbls]

        for l, t in zip(lbls, _labels):
            l.set_text(t)
    return cs
