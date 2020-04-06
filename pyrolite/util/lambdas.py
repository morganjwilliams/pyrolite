import numpy as np
from sympy.solvers.solvers import nsolve
from sympy import symbols, var
from functools import partial
import scipy
from .meta import update_docstring_references
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

@update_docstring_references
def OP_constants(xs, degree=3, rounding=None, tol=10 ** -14):
    r"""
    Finds the parameters
    :math:`(\beta_0), (\gamma_0, \gamma_1), (\delta_0, \delta_1, \delta_2)` etc.
    for constructing orthogonal polynomial functions `f(x)` over a fixed set of values
    of independent variable `x`.
    Used for obtaining lambda values for dimensional reduction of REE data [#ref_1]_.

    Parameters
    ----------
    xs : :class:`numpy.ndarray`
        Indexes over which to generate the orthogonal polynomials.
    degree : :class:`int`
        Maximum polynomial degree. E.g. 2 will generate constant, linear, and quadratic
        polynomial components.
    tol : :class:`float`
        Convergence tolerance for solver.
    rounding : :class:`int`
        Precision for the orthogonal polynomial coefficents.

    Returns
    ---------
    :class:`list`
        List of tuples corresponding to coefficients for each of the polynomial
        components. I.e the first tuple will be empty, the second will contain a single
        coefficient etc.

    Notes
    -----
        Parameters are used to construct orthogonal polymomials of the general form:

        .. math::

            f(x) &= a_0 \\
            &+ a_1 * (x - \beta) \\
            &+ a_2 * (x - \gamma_0) * (x - \gamma_1) \\
            &+ a_3 * (x - \delta_0) * (x - \delta_1) * (x - \delta_2) \\

    See Also
    ---------
    :func:`~pyrolite.util.math.lambdas`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    References
    -----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__
    """
    xs = np.array(xs)
    x = var("x")
    params = []
    for d in range(degree):
        ps = symbols("{}0:{}".format(chr(945 + d), d))
        logger.debug("Generating {} DIM {} equations for {}.".format(d, d, ps))
        if d:
            eqs = []
            for _deg in range(d):
                q = 1
                if _deg:
                    q = x ** _deg
                for p in ps:
                    q *= x - p
                eqs.append(q)
            sums = []
            for q in eqs:
                sumq = 0.0
                for xi in xs:
                    sumq += q.subs(dict(x=xi))
                sums.append(sumq)

            guess = np.linspace(np.nanmin(xs), np.nanmax(xs), d + 2)[1:-1]
            result = nsolve(sums, ps, list(guess), tol=tol)
            if rounding is not None:
                result = np.around(np.array(result, dtype=np.float), decimals=rounding)
            params.append(tuple(result))
        else:
            params.append(())  # first parameter
    return params


def lambda_poly(x, ps):
    """
    Polynomial lambda_n(x) given parameters ps with len(ps) = n.

    Parameters
    -----------
    x : :class:`numpy.ndarray`
        X values to calculate the function at.
    ps: :class:`tuple`
        Parameter set tuple. E.g. parameters `(a, b)` from :math:`f(x) = (x-a)(x-b)`.

    Returns
    --------
    :class:`numpy.ndarray`
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    result = np.ones(len(x))
    for p in ps:
        result = result * (x - p)
    return result.astype(np.float)


def lambda_min_func(ls, ys, arrs, power=2.0):
    """
    Cost function for lambda optitmization.

    Parameters
    ------------
    ls : :class:`numpy.ndarray`
        Lambda values, effectively weights for the polynomial components.
    ys : :class:`numpy.ndarray`
        Target y values.
    arrs : :class:`numpy.ndarray`
        Arrays representing the individual unweighted orthaogonal polynomial components.
        E.g. arrs[0] = `[a, a, a]`, arrs[1] = `[(x-b), (x-b), (x-b)]` etc.
    power : :class:`float`
        Power for the cost function. 1 for MAE/L1 norm, 2 for MSD/L2 norm.

    Returns
    -------
    :class:`numpy.ndarray`
        Cost at the given set of `ls`.

    Todo
    -----
        * Rewrite cost function for readability with :func:`lambdas`.
    """
    cost = np.abs(np.dot(ls, arrs) - ys) ** power
    cost[np.isnan(cost)] = 0.0  # can't change nans - don't penalise them
    return cost


@update_docstring_references
def lambdas(
    arr: np.ndarray,
    xs=np.array([]),
    params=None,
    guess=None,
    degree=5,
    costf_power=2.0,
    residuals=False,
    min_func=lambda_min_func,
):
    """
    Parameterises values based on linear combination of orthogonal polynomials
    over a given set of values for independent variable `x`. [#ref_1]_

    Parameters
    -----------
    arr : :class:`numpy.ndarray`
        Target data to fit. For geochemical data, this is typically normalised
        so we can fit a smooth function.
    xs : :class:`numpy.ndarray`
        Values of `x` to construct the polymomials over.
    params : :class:`list`, :code:`None`
        Orthogonal polynomial coefficients (see :func:`OP_constants`). Defaults to
        `None`, in which case these coefficinets are generated automatically.
    guess : :class:`numpy.ndarray`
        Starting for values of lambdas. Used as starting point for optimization.
    degree : :class:`int`
        Maximum degree polymomial component to include.
    costf_power : :class:`float`
        Power of the optimization cost function.
    residuals : :class:`bool`
        Whether to return residuals with the optimized results.
    min_func : :class:`Callable`
        Cost function to use for optimization of lambdas.

    Returns
    --------
    :class:`numpy.ndarray` | (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        Optimial results for weights of orthogonal polymomial regression (`lambdas`).

    See Also
    ---------
    :func:`~pyrolite.util.math.OP_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    Todo
    -----
        * Change the cost function such that the power is controlled externally

    References
    -----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__
    """
    if np.isnan(arr).any():  # With missing data, the method can't be used.
        x = np.nan * np.ones(degree)
        res = np.nan * np.ones(degree)
    else:
        guess = guess or np.exp(np.arange(degree) + 2)
        params = params or OP_constants(xs, degree=degree)
        # arrays representing the unweighted individual polynomial components
        fs = np.array([lambda_poly(xs, pset) for pset in params])
        result = scipy.optimize.least_squares(
            min_func, guess, args=(arr, fs, costf_power)
        )
        x = result.x
        res = result.fun
    if residuals:
        return x, res
    else:
        return x


def lambda_poly_func(lambdas: np.ndarray, params=None, pxs=None, degree=5):
    """
    Expansion of lambda parameters back to the original space. Returns a
    function which evaluates the sum of the orthogonal polynomials at given
    `x` values.

    Parameters
    ------------
    lambdas: :class:`numpy.ndarray`
        Lambda values to weight combination of polynomials.
    params: :class:`list`(:class:`tuple`)
        Parameters for the orthogonal polynomial decomposition.
    pxs: :class:`numpy.ndarray`
        x values used to construct the lambda values. [#note_1]_
    degree: :class:`int`
        Degree of the orthogonal polynomial decomposition. [#note_1]_

    See Also
    ---------
    :func:`~pyrolite.util.math.lambdas`
    :func:`~pyrolite.util.math.OP_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    Notes
    -----
        .. [#note_1] Only needed if parameters are not supplied
    """
    if params is None and pxs is not None:
        params = OP_constants(pxs, degree=degree)
    elif params is None and pxs is None:
        msg = """Must provide either x values to construct parameters,
                 or the parameters themselves."""
        raise AssertionError(msg)

    def lambda_poly_f(xarr):
        """
        Calculates the sum of decomposed polynomial components
        at given x values.

        Parameters
        -----------
        xarr: :class:`numpy.ndarray`
            X values at which to evaluate the function.
        """
        arrs = np.array([lambda_poly(xarr, pset) for pset in params])
        return np.dot(lambdas, arrs)

    return lambda_poly_f
