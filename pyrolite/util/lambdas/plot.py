"""
Functions for the visualisation of reconstructed and deconstructed parameterised REE
profiles based on parameterisations using 'lambdas' (and tetrad-equivalent weights
'taus').
"""

import matplotlib.axes
import numpy as np
import pandas as pd

from ... import plot
from ...geochem.ind import REE, get_ionic_radii
from ..log import Handle
from .eval import (
    get_function_components,
    get_lambda_poly_function,
    get_tetrads_function,
)
from .params import _get_params
from .transform import REE_radii_to_z, REE_z_to_radii

logger = Handle(__name__)


def plot_lambdas_components(
    lambdas: pd.Series | np.ndarray,
    params: list[tuple[float, ...]] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot a decomposed orthogonal polynomial from a single set of lambda coefficients.

    Parameters
    ----------
    lambdas
        1D array of lambdas.
    params : list
        List of orthongonal polynomial parameters, if defaults are not used.
    ax : matplotlib.axes.Axes
        Optionally specified axes to plot on.
    index : str
        Index to use for the plot (one of `"index", "radii", "z"`).

    Returns
    --------
    matplotlib.axes.Axes
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
            rf"$r^{len(p)}: \lambda_{len(p)}"
            + [rf"\cdot f_{len(p)}", ""][int(len(p) == 0)]
            + "$"
        )
        ax.plot(xs, l_func(xs), label=label, ls="--", **kwargs)  # plot the polynomials
    return ax


def plot_tetrads_components(
    taus: pd.Series | np.ndarray,
    tetrad_params: list[tuple[float, ...]] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    index: str = "radii",
    logy: bool = True,
    drop0: bool = True,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Individually plot the four tetrad components for one set of $\tau$s.

    Parameters
    ----------
    taus : numpy.ndarray
        1D array of $\tau$ tetrad function coefficients.
    tetrad_params : list
        List of tetrad parameters, if defaults are not used.
    ax : matplotlib.axes.Axes
        Optionally specified axes to plot on.
    index : str
        Index to use for the plot (one of `"index", "radii", "z"`).
    logy : bool
        Whether to log-scale the y-axis.
    drop0 : bool
        Whether to remove zeroes from the outputs such that individual tetrad
        functions are shown only within their respective bounds (and not across the
        entire REE, where their effective values are zero).
    """
    # flat 1D array of ts
    f = get_tetrads_function(params=tetrad_params)

    z = np.arange(57, 72)  # marker
    linez = np.linspace(57, 71, 1000)  # line

    taus = taus.reshape(-1, 1)
    ys = (taus * f(z, sum_tetrads=False)).squeeze()
    liney = (taus * f(linez, sum_tetrads=False)).squeeze()

    REE_z_to_radii(z)
    REE_z_to_radii(linez)
    ####################################################################################
    if index in ["radii", "elements"]:
        ax = plot.spider.REE_v_radii(logy=logy, index=index, ax=ax, **kwargs)
    else:
        index = "z"
        ax = plot.spider.spider(
            np.array([np.nan] * len(z)), indexes=z, logy=logy, ax=ax, **kwargs
        )
        ax.set_xticklabels(REE(dropPm=False))
        # xs = z
        # linex = linez

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
    coefficients: pd.Series | np.ndarray,
    tetrads: bool = False,
    params: list[tuple[float, ...]] | None = None,
    tetrad_params: list[tuple[float, ...]] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    index: str = "radii",
    logy: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    r"""
    Plot the reconstructed REE profiles of a 2D dataset of coefficients ($\lambda$s,
    and optionally $\tau$s).

    Parameters
    ----------
    coefficients : numpy.ndarray
        2D array of $\lambda$ orthogonal polynomial coefficients, and optionally
        including $\tau$ tetrad function coefficients in the last four columns
        (where `tetrads=True`).
    tetrads : bool
        Whether the coefficient array contains tetrad coefficients ($\tau$s).
    params : list
        List of orthongonal polynomial parameters, if defaults are not used.
    tetrad_params : list
        List of tetrad parameters, if defaults are not used.
    ax : matplotlib.axes.Axes
        Optionally specified axes to plot on.
    index : str
        Index to use for the plot (one of `"index", "radii", "z"`).
    logy : bool
        Whether to log-scale the y-axis.

    Returns
    --------
    matplotlib.axes.Axes
    """
    radii = get_ionic_radii(REE(), charge=3, coordination=8)
    # check the degree required for the lambda coefficients and get the OP parameters
    lambda_degree = coefficients.shape[1] - [0, 4][tetrads]
    params = _get_params(params or "full", degree=lambda_degree)

    # get the components and y values for the points/element locations
    _, x0, components = get_function_components(
        radii,
        params=params,
        fit_tetrads=tetrads,
        tetrad_params=tetrad_params,
    )
    ys = np.exp(coefficients @ components)
    # get the components and y values for the smooth lines
    lineradii = np.linspace(radii[0], radii[-1], 1000)

    _, _x0, linecomponents = get_function_components(
        lineradii,
        params=params,
        fit_tetrads=tetrads,
        tetrad_params=tetrad_params,
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
        **{**kwargs, "marker": ""},
    )

    return ax
