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


def parse_sigmas(y, sigmas=None):
    """
    Disambigaute a value or set of sigmas for a dataset for use in lambda-fitting
    algorithms.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        2D array of y values.
    sigmas : :class:`float` | :class:`numpy.ndarray`
        Single value or 1D array of sigmas.

    Returns
    -------
    sigmas : :class:`float` | :class:`numpy.ndarray`
        Single value or 1D array of sigmas.

    Notes
    -----
    If no sigmas are provided, 1% of the mean y values will be returned.
    """
    # if sigmas is none, it's assumed 2% uncertainty on log-transformed
    # normalised abundances; 1D and 2D arrays should also work
    sigma2d = False
    if sigmas is None:
        sigmas = 0.01 * y.mean(axis=0)
    elif isinstance(sigmas, float):
        sigmas = sigmas * y.mean(axis=0)
    elif sigmas.ndim > 1:
        if any(ix == 1 for ix in sigmas.shape):
            sigmas = sigmas.flatten()
        else:
            msg = "2D uncertainty estimation not yet implemented."
            raise NotImplementedError(msg)
    else:
        pass  # should be a 1D array

    return sigmas


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

    Returns
    -------
    B, se : :class:`numpy.ndarray`
        Arrays for the optimized parameter values (B) and parameter
        uncertaintes (se, 1σ).
    """
    X = np.array(func_components)

    md_inds, patterns = md_pattern(y)
    xd = len(func_components)
    # for each missing data pattern, we create the matrix A - rather than each row
    sigmas = parse_sigmas(y, sigmas=sigmas)

    B = np.ones((y.shape[0], len(func_components))) * np.nan
    se = np.ones((y.shape[0], len(func_components))) * np.nan
    x2 = np.ones((y.shape[0], 1)) * np.nan

    for ind in np.unique(md_inds):
        row_fltr = md_inds == ind  # rows with this pattern
        missing_fltr = ~patterns[ind]["pattern"]  # boolean presence-absence filter
        if missing_fltr.sum():  # ignore completely empty row
            yd = missing_fltr.sum()
            # underscores for local variables referring to part of the array
            _x, _y, _sigmas = (
                X[:, missing_fltr],
                y[np.ix_(row_fltr, missing_fltr)],
                sigmas[missing_fltr],
            )
            invXX = np.linalg.pinv(_x @ _x.T)

            _B = (invXX @ _x @ _y.T).T  # parameter estimates

            ############################################################################
            # Uncertainties following Bevington
            est = (_x.T @ _B.T).T  # modelled values
            res = _y - est  # residuals over all rows
            chi_squared = (res ** 2 / sigmas ** 2).sum(axis=1)  # chi_squared per row
            mse = (res ** 2).sum(axis=1)
            # mse per row divided by degrees of freedom
            _se = np.sqrt(mse[:, None] / (yd - xd) * np.diag(invXX)[None, :])

            """
            # row matrix beta = sum (1/ sigmas**2 * yi * fk(sxi))
            β = np.array(
                [
                    ((fk[missing_fltr] * _y) / _sigmas ** 2).sum(axis=1)
                    for fk in X
                ]
            ).T

            # if the sigmas are 2D, this will need to be per-row (i.e. ϵ varies by row)
            α = (X / sigmas[missing_fltr] ** 2) @ X.T
            ϵ = np.linalg.inv(α)
            a = β @ ϵ

            _se = np.sqrt(mse[:, None] / (yd - xd) * np.diag(ϵ)[None, :])
            """
            B[row_fltr, :] = _B
            se[row_fltr, :] = _se
            x2[row_fltr, :] = chi_squared[:, None]
    return B, se, x2


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

    Returns
    -------
    arr, uarr : :class:`numpy.ndarray`
        Arrays for the optimized parameter values (arr) and parameter
        uncertaintes (uarr, 1σ).

    """
    m, n = y.shape[0], x0.size  # shape of output
    sigmas = parse_sigmas(y, sigmas=sigmas)
    B = np.ones((y.shape[0], len(func_components))) * np.nan
    se = np.ones((y.shape[0], len(func_components))) * np.nan
    x2 = np.ones((y.shape[0], 1)) * np.nan
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

        chi_squared = (res.fun ** 2 / sigmas ** 2).sum()
        B[row, :] = res.x
        se[row, :] = np.sqrt(np.diag(pcov))  # sigmas on parameters
        x2[row, 0] = chi_squared
    return B, se, x2


@update_docstring_references
def lambdas_optimize(
    df: pd.DataFrame,
    radii,
    params=None,
    guess=None,
    fit_tetrads=False,
    tetrad_params=None,
    fit_method="opt",
    add_SE=False,
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
    add_SE : :class:`bool`
        Append parameter standard errors to the dataframe.

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

    B, se, X2 = fit(df.pyrochem.REE.values, np.array(x0), func_components, **kwargs)

    lambdas = pd.DataFrame(
        B,
        index=df.index,
        columns=names,
        dtype="float32",
    )
    if add_SE:
        lambdas.loc[:, [n + "_" + chr(963) for n in names]] = se
    if add_X2:
        lambdas["X2"] = X2
    return lambdas
