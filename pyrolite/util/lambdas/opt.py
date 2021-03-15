"""
Functions for optimization-based REE profile fitting and parameter uncertainty
estimation.
"""
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.linalg
from ..meta import update_docstring_references
from ..missing import md_pattern
from .eval import get_function_components
from .params import parse_sigmas
from ..log import Handle

logger = Handle(__name__)


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


def linear_fit_components(y, x0, func_components, sigmas=None):
    """
    Fit a weighted sum of function components using linear algebra.

    Parameters
    -----------
    y : :class:`numpy.ndarray`
        Array of target values to fit.
    x0 : :class:`numpy.ndarray`
        Starting guess for the function weights.
    func_components : :class:`list` ( :class:`numpy.ndarray` )
        List of arrays representing static/evaluated function components.
    sigmas : :class:`float` | :class:`numpy.ndarray`
        Single value or 1D array of normalised observed value uncertainties
        (:math:`\sigma_{REE} / REE`).

    Returns
    -------
    B, s, χ2 : :class:`numpy.ndarray`
        Arrays for the optimized parameter values (B; (n, d)), parameter
        uncertaintes (s, 1σ; (n, d)) and chi-chi_squared (χ2; (n, 1)).
    """
    X = np.array(func_components).T  # components as vectors [1 f0 f1 f2]
    xd = len(func_components)  # number of parameters

    sigmas = parse_sigmas(y, sigmas=sigmas)

    B = np.ones((y.shape[0], len(func_components))) * np.nan
    s = np.ones((y.shape[0], len(func_components))) * np.nan
    χ2 = np.ones((y.shape[0], 1)) * np.nan

    # for each missing data pattern, we create the matrix A - rather than each row
    md_inds, patterns = md_pattern(y)
    for ind in np.unique(md_inds):
        row_fltr = md_inds == ind  # rows with this pattern
        missing_fltr = ~patterns[ind]["pattern"]  # boolean presence-absence filter
        if missing_fltr.sum():  # ignore completely empty row
            yd = missing_fltr.sum()
            # underscores for local variables referring to part of the array
            _x, _y, _sigmas = (
                X[missing_fltr, :],
                y[np.ix_(row_fltr, missing_fltr)],
                sigmas[missing_fltr],
            )
            # weights derived from reciprocal variance of y-uncertaintes
            # W = np.eye(_sigmas.shape[0]) * 1 / _sigmas ** 2  # weights
            # assuming the errors on y are uncorrelated, we can use _sigmas as
            # whitening transformation for X and y:
            # w = np.diag(np.sqrt(W))
            w = 1 / _sigmas
            _x *= w.reshape(-1, 1)
            _y *= w

            invXX = np.linalg.inv(_x.T @ _x)
            _B = (invXX @ _x.T @ _y.T).T  # parameter estimates
            ############################################################################
            est = (_x @ _B.T).T  # estimated values of y
            residuals = _y - est  # residuals
            S = (residuals ** 2).sum(axis=1)  # residual sum of squares
            # H = X @ invXWX @ X.T @ W  # Hat matrix
            dof = yd - xd  # effective degrees of freedom (for this mising filter)
            # calculate the reduced_chi_squared per row (divided by degrees of freedom)
            # note due to the whitening transformation above, S is effectively
            # sum(residual/sigma)**2
            reduced_chi_squared = S / dof
            _s = np.sqrt(reduced_chi_squared.reshape(-1, 1) * np.diag(invXX))

            B[row_fltr, :] = _B
            s[row_fltr, :] = _s
            χ2[row_fltr, 0] = reduced_chi_squared
    return B, s, χ2


def optimize_fit_components(
    y, x0, func_components, residuals_function=_residuals_func, sigmas=None
):
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
    sigmas : :class:`float` | :class:`numpy.ndarray`
        Single value or 1D array of normalised observed value uncertainties
        (:math:`\sigma_{REE} / REE`).

    Returns
    -------
    B, s, χ2 : :class:`numpy.ndarray`
        Arrays for the optimized parameter values (B; (n, d)), parameter
        uncertaintes (s, 1σ; (n, d)) and chi-chi_squared (χ2; (n, 1)).
    """
    m, n = y.shape[0], x0.size  # shape of output
    sigmas = parse_sigmas(y, sigmas=sigmas)
    B = np.ones((y.shape[0], len(func_components))) * np.nan
    s = np.ones((y.shape[0], len(func_components))) * np.nan
    χ2 = np.ones((y.shape[0], 1)) * np.nan
    for row in range(m):
        res = scipy.optimize.least_squares(
            residuals_function,
            x0,
            args=(
                y[row, :],
                func_components,
            ),
        )
        # get the covariance matrix of the parameters from the jacobian
        pcov = pcov_from_jac(res.jac)
        yd, xd = y.shape[1], x0.size
        if yd > xd:  # check samples in y vs parameter dimension
            s_sq = res.cost / (yd - xd)
            pcov = pcov * s_sq
        else:
            pcov.fill(np.inf)

        dof = yd - xd  # effective degrees of freedom
        reduced_chi_squared = (res.fun ** 2 / sigmas ** 2).sum() / dof
        B[row, :] = res.x
        s[row, :] = np.sqrt(np.diag(pcov))  # sigmas on parameters
        χ2[row, 0] = reduced_chi_squared
    return B, s, χ2


@update_docstring_references
def lambdas_optimize(
    df: pd.DataFrame,
    radii,
    params=None,
    guess=None,
    fit_tetrads=False,
    tetrad_params=None,
    fit_method="opt",
    sigmas=None,
    add_uncertainties=False,
    add_X2=False,
    **kwargs
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
    fit_method : :class:`str`
        Which fit method to use: :code:`"optimization"` or :code:`"linear"`.
    sigmas : :class:`float` | :class:`numpy.ndarray`
        Single value or 1D array of observed value uncertainties.
    add_uncertainties : :class:`bool`
        Whether to append estimated parameter uncertainties to the dataframe.
    add_X2 : :class:`bool`
        Whether to append the chi-squared values (χ2) to the dataframe.

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
    if fit_method.lower().startswith("opt"):
        fit = optimize_fit_components
    else:
        fit = linear_fit_components

    B, s, χ2 = fit(
        df.pyrochem.REE.values, np.array(x0), func_components, sigmas=sigmas, **kwargs
    )

    lambdas = pd.DataFrame(
        B,
        index=df.index,
        columns=names,
        dtype="float32",
    )
    if add_uncertainties:
        lambdas.loc[:, [n + "_" + chr(963) for n in names]] = s
    if add_X2:
        lambdas["X2"] = χ2
    return lambdas
