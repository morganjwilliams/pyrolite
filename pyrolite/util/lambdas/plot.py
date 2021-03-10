"""
Functions for the visualisation of reconstructed and deconstructed parameterised REE
profiles based on parameterisations using 'lambdas' (and tetrad-equivalent weights
'taus').
"""
import numpy as np
from ... import plot
from ...geochem.ind import get_ionic_radii, REE
from .params import orthogonal_polynomial_constants, _get_params
from .eval import (
    get_lambda_poly_function,
    get_tetrads_function,
    get_function_components,
)
from .transform import REE_z_to_radii, REE_radii_to_z
from ..log import Handle

logger = Handle(__name__)

def plot_lambdas_components(lambdas, params=None, ax=None, **kwargs):
    """
    Plot a decomposed orthogonal polynomial from a single set of lambda coefficients.

    Parameters
    ----------
    lambdas
        1D array of lambdas.
    params : :class:`list`
        List of orthongonal polynomial parameters, if defaults are not used.
    ax : :class:`matplotlib.axes.Axes`
        Optionally specified axes to plot on.
    index : :class:`str`
        Index to use for the plot (one of :code:`"index", "radii", "z"`).

    Returns
    --------
    :class:`matplotlib.axes.Axes`
    """
    degree = lambdas.size
    params = _get_params(params=params, degree=degree)
    # check the degree and parameters are of consistent degree?
    reconstructed_func = get_lambda_poly_function(lambdas, params)

    ax = plot.spider.REE_v_radii(ax=ax)

    radii = np.array(get_ionic_radii(REE(), charge=3, coordination=8))
    xs = np.linspace(np.max(radii), np.min(radii), 100)
    ax.plot(xs, reconstructed_func(xs), label="Regression", color="k", **kwargs)
    for w, p in zip(lambdas, params):  # plot the components
        l_func = get_lambda_poly_function(
            [w], [p]
        )  # pasing singluar vaules and one tuple
        label = (
            r"$r^{}: \lambda_{}".format(len(p), len(p))
            + [r"\cdot f_{}".format(len(p)), ""][int(len(p) == 0)]
            + "$"
        )
        ax.plot(xs, l_func(xs), label=label, ls="--", **kwargs)  # plot the polynomials
    return ax


def plot_tetrads_components(
    taus, tetrad_params=None, ax=None, index="radii", logy=True, drop0=True, **kwargs
):
    """
    Individually plot the four tetrad components for one set of $\tau$s.

    Parameters
    ----------
    taus : :class:`numpy.ndarray`
        1D array of $\tau$ tetrad function coefficients.
    tetrad_params : :class:`list`
        List of tetrad parameters, if defaults are not used.
    ax : :class:`matplotlib.axes.Axes`
        Optionally specified axes to plot on.
    index : :class:`str`
        Index to use for the plot (one of :code:`"index", "radii", "z"`).
    logy : :class:`bool`
        Whether to log-scale the y-axis.
    drop0 : :class:`bool`
        Whether to remove zeroes from the outputs such that individual tetrad
        functions are shown only within their respective bounds (and not across the
        entire REE, where their effective values are zero).
    """
    # flat 1D array of ts
    f = get_tetrads_function(params=tetrad_params)

    z = np.arange(57, 72)  # marker
    linez = np.linspace(57, 71, 1000)  # line

    taus = taus.reshape(-1, 1)
    ys = (taus * f(z, sum=False)).squeeze()
    liney = (taus * f(linez, sum=False)).squeeze()

    xs = REE_z_to_radii(z)
    linex = REE_z_to_radii(linez)
    ####################################################################################
    if index in ["radii", "elements"]:
        ax = plot.spider.REE_v_radii(logy=logy, index=index, ax=ax, **kwargs)
    else:
        index = "z"
        ax = plot.spider.spider(
            np.array([np.nan] * len(z)), indexes=z, logy=logy, ax=ax, **kwargs
        )
        ax.set_xticklabels(REE(dropPm=False))
        xs = z
        linex = linez

    if drop0:
        yfltr = np.isclose(ys, 0)
        # we can leave in markers which should actually be there at zero - 1/ea tetrad
        yfltr = yfltr * (
            1 - np.isclose(z[:, None] - np.array([57, 64, 64, 71]).T, 0).T
        ).astype(bool)
        ys[yfltr] = np.nan
        liney[np.isclose(liney, 0)] = np.nan
    return ax


def plot_profiles(
    coefficients,
    tetrads=False,
    params=None,
    tetrad_params=None,
    ax=None,
    index="radii",
    logy=False,
    **kwargs
):
    """
    Plot the reconstructed REE profiles of a 2D dataset of coefficients ($\lambda$s,
    and optionally $\tau$s).

    Parameters
    ----------
    coefficients : :class:`numpy.ndarray`
        2D array of $\lambda$ orthogonal polynomial coefficients, and optionally
        including $\tau$ tetrad function coefficients in the last four columns
        (where :code:`tetrads=True`).
    tetrads : :class:`bool`
        Whether the coefficient array contains tetrad coefficients ($\tau$s).
    params : :class:`list`
        List of orthongonal polynomial parameters, if defaults are not used.
    tetrad_params : :class:`list`
        List of tetrad parameters, if defaults are not used.
    ax : :class:`matplotlib.axes.Axes`
        Optionally specified axes to plot on.
    index : :class:`str`
        Index to use for the plot (one of :code:`"index", "radii", "z"`).
    logy : :class:`bool`
        Whether to log-scale the y-axis.

    Returns
    --------
    :class:`matplotlib.axes.Axes`
    """
    radii = get_ionic_radii(REE(), charge=3, coordination=8)
    # check the degree required for the lambda coefficients and get the OP parameters
    lambda_degree = coefficients.shape[1] - [0, 4][tetrads]
    params = _get_params(params or "full", degree=lambda_degree)

    # get the components and y values for the points/element locations
    names, x0, components = get_function_components(
        radii, params=params, fit_tetrads=tetrads, tetrad_params=tetrad_params,
    )
    ys = np.exp(coefficients @ components)
    # get the components and y values for the smooth lines
    lineradii = np.linspace(radii[0], radii[-1], 1000)

    names, x0, linecomponents = get_function_components(
        lineradii, params=params, fit_tetrads=tetrads, tetrad_params=tetrad_params,
    )
    liney = np.exp(coefficients @ linecomponents)
    z, linez = REE_radii_to_z(radii), REE_radii_to_z(lineradii)
    xs, linex = radii, lineradii
    ####################################################################################
    if index in ["radii", "elements"]:
        ax = plot.spider.REE_v_radii(ax=ax, logy=logy, index=index, **kwargs)
    else:
        index = "z"
        ax = plot.spider.spider(
            np.array([np.nan] * len(z)), ax=ax, indexes=z, logy=logy, **kwargs
        )
        ax.set_xticklabels(REE(dropPm=False))
        xs = z
        linex = linez

    # ys = np.exp(ys)
    # liney = np.exp(liney)
    # scatter-only spider
    plot.spider.spider(
        ys, ax=ax, indexes=xs, logy=logy, set_ticks=False, **{**kwargs, "linewidth": 0}
    )
    # line-only spider
    plot.spider.spider(
        liney,
        ax=ax,
        indexes=linex,
        logy=logy,
        set_ticks=False,
        **{**kwargs, "marker": ""}
    )

    return ax
