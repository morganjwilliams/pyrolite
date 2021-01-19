"""
Functions for optimization-based REE profile fitting and parameter uncertainty
estimation.
"""
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.linalg
from ..meta import update_docstring_references
from ..log import Handle
from .eval import get_function_components

logger = Handle(__file__)


def _cost_func(ls, ys, func_components, power=1.0):
    """
    Cost function for lambda optimization.

    Parameters
    ------------
    ls : :class:`numpy.ndarray`
        Lambda values, effectively weights for the polynomial components.
    ys : :class:`numpy.ndarray`
        Target y values.
    func_components : :class:`numpy.ndarray`
        Arrays representing the individual unweighted function components.
        E.g. :code:`[[a, a, ...], [x - b, x - b, ...], ...]` for lambdas.
    power : :class:`float`
        Power for the cost function.

    Returns
    -------
    :class:`numpy.ndarray`
        Cost at the given set of `ls`.
    """
    cost = np.abs(ls @ func_components - ys) ** power
    cost[np.isnan(cost)] = 0.0  # can't change nans - don't penalise them
    return cost


def _residuals_func(ls, ys, func_components):
    """
    Residuals function for lambda optimization.

    Parameters
    ------------
    ls : :class:`numpy.ndarray`
        Lambda values, effectively weights for the polynomial components.
    ys : :class:`numpy.ndarray`
        Target y values.
    func_components : :class:`numpy.ndarray`
        Arrays representing the individual unweighted unweighted function components.
        E.g. :code:`[[a, a, ...], [x - b, x - b, ...], ...]` for lambdas.

    Returns
    -------
    :class:`numpy.ndarray`
        Residuals at the given set of `ls`.
    """
    res = ls @ func_components - ys
    res[np.isnan(res)] = 0.0  # can't change nans - don't penalise them
    return res


def pcov_from_jac(jac):
    """
    Extract a covariance matrix from a Jacobian matrix returned from
    :mod:`scipy.optimize` functions.

    Parameters
    ----------
    jac : :class:`numpy.ndarray`
        Jacobian array.

    Returns
    -------
    pcov : :class:`numpy.ndarray`
        Square covariance array; this hasn't yet been scaled by residuals.
    """
    # from scipy.opt minpack
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = scipy.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    return pcov


def fit_components(y, x0, func_components, residuals_function=_residuals_func):
    """
    Fit a weighted sum of function components using
    :func:`scipy.optimize.least_squares`.

    Parameters
    -----------
    y : :class:`numpy.ndarray`
        Array of target values to fit.
    x0 : :class:`numpy.ndarray`
        Starting guess for the function weights.
    func_components : :class:`list` ( :class:`numpy.ndarray` )
        List of arrays representing static/evaluated function components.
    redsiduals_function : callable
        Callable funciton to compute residuals which accepts ordered arguments for
        weights, target values and function components.

    Returns
    -------
    arr, uarr : :class:`numpy.ndarray`
        Arrays for the optimized parameter values (arr) and parameter
        uncertaintes (uarr, 1σ).
    """
    m, n = y.shape[0], x0.size  # shape of output
    arr = np.ones((m, n)) * np.nan
    uarr = np.ones((m, n)) * np.nan
    for row in range(m):
        res = scipy.optimize.least_squares(
            residuals_function,
            x0,
            args=(
                y[row, :],
                func_components,
            ),
        )
        arr[row, :] = res.x
        # get the covariance matrix of the parameters from the jacobian
        pcov = pcov_from_jac(res.jac)
        yd, xd = y.shape[0], x0.size
        if yd > xd:  # check samples in y vs paramater dimension
            s_sq = res.cost / (yd - xd)
            pcov = pcov * s_sq
        else:
            pcov.fill(np.inf)
        uarr[row, :] = np.sqrt(np.diag(pcov))  # sigmas on parameters
    return arr, uarr


@update_docstring_references
def lambdas_optimize(
    df: pd.DataFrame,
    radii,
    params=None,
    guess=None,
    fit_tetrads=False,
    tetrad_params=None,
    residuals_function=_residuals_func,
):
    """
    Parameterises values based on linear combination of orthogonal polynomials
    over a given set of values for independent variable `x`. [#ref_1]_

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Target data to fit. For geochemical data, this is typically normalised
        so we can fit a smooth function.
    radii : :class:`list`, :class:`numpy.ndarray`
        Radii at which to evaluate the orthogonal polynomial.
    params : :class:`list`, :code:`None`
        Orthogonal polynomial coefficients (see
        :func:`orthogonal_polynomial_constants`).
    fit_tetrads : :class:`bool`
        Whether to also fit the patterns for tetrads.
    tetrad_params : :class:`list`
        List of parameter sets for tetrad functions.
    residuals_function : :class:`Callable`
        Residuals function to use for optimization of lambdas.

    Returns
    --------
    :class:`numpy.ndarray` | (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        Optimial results for weights of orthogonal polymomial regression (`lambdas`).

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.params.orthogonal_polynomial_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    References
    ----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__
    """
    assert params is not None
    degree = len(params)
    # arrays representing the unweighted individual polynomial components
    names, x0, func_components = get_function_components(
        radii, params=params, fit_tetrads=fit_tetrads, tetrad_params=tetrad_params
    )
    arr, uarr = fit_components(df.pyrochem.REE.values, np.array(x0), func_components)
    lambdas = pd.DataFrame(
        arr,
        index=df.index,
        columns=names,
        dtype="float32",
    )
    return lambdas
