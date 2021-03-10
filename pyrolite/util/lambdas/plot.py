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


def plot_lambdas_components(lambdas, ax=None, params=None, degree=4, **kwargs):
    """
    Plot a decomposed orthogonal polynomial using the lambda coefficients.

    Parameters
    ----------
    lambdas
        1D array of lambdas.
    ax : :class:`matplotlib.axes.Axes`
        Axis to plot on.

    Returns
    --------
    :class:`matplotlib.axes.Axes`
    """

    params = _get_params(params=params, degree=degree)
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
            "$r^{}: \lambda_{}".format(len(p), len(p))
            + ["\cdot f_{}".format(len(p)), ""][int(len(p) == 0)]
            + "$"
        )
        ax.plot(xs, l_func(xs), label=label, ls="--", **kwargs)  # plot the polynomials
    return ax


def plot_lambdas_profiles(lambdas, ax=None, params=None, degree=4, **kwargs):
    """
    Plot the REE patterns reconstructed from lambdas.

    Parameters
    ----------
    lambdas
        2D array of lambdas.
    ax : :class:`matplotlib.axes.Axes`
        Axis to plot on.

    Returns
    --------
    :class:`matplotlib.axes.Axes`
    """
    params = _get_params(params=params, degree=degree)
    reconstructed_func = get_lambda_poly_function(lambdas, params)

    ax = plot.spider.REE_v_radii(ax=ax)

    radii = np.array(get_ionic_radii(REE(), charge=3, coordination=8))
    xs = np.linspace(np.max(radii), np.min(radii), 100)
    ax.plot(xs, reconstructed_func(xs), label="Regression", color="k", **kwargs)
    return ax


def plot_tetrads_components():
    # 1D array of ts
    f = get_tetrads_function(params=None)

    z = np.arange(57, 72)  # marker
    linez = np.linspace(57, 71, 1000)  # line

    ts = np.atleast_2d(ts)
    ys = (ts * f(z, sum=False)).squeeze()
    liney = (ts * f(linez, sum=False)).squeeze()

    xs = REE_z_to_radii(z)
    linex = REE_z_to_radii(linez)
    ####################################################################################
    if index in ["radii", "elements"]:
        ax = plot.spider.REE_v_radii(logy=logy, index=index, **kwargs)
    else:
        index = "z"
        ax = plot.spider.spider(
            np.array([np.nan] * len(z)), indexes=z, logy=logy, **kwargs
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


def plot_profiles(
    lambdas,
    tetrads=False,
    params=None,
    tetrad_params=None,
    index="radii",
    logy=False,
    ax=None,
    **kwargs
):
    """
    Plot the tetrad-only profiles of REE patterns.
    """
    radii = get_ionic_radii(REE(), charge=3, coordination=8)
    names, x0, components = get_function_components(
        radii,
        params=_get_params(params or "full", degree=5),
        fit_tetrads=tetrads,
        tetrad_params=tetrad_params,
    )
    ys = np.exp(lambdas @ components)
    lineradii = np.linspace(radii[0], radii[-1], 1000)
    names, x0, linecomponents = get_function_components(
        lineradii,
        params=_get_params(params or "full", degree=5),
        fit_tetrads=tetrads,
        tetrad_params=tetrad_params,
    )
    liney = np.exp(lambdas @ linecomponents)
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
