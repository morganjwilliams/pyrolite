import numpy as np
from sympy.solvers.solvers import nsolve
from sympy import symbols, var
from functools import partial
from scipy import optimize
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def on_finite(arr, f):
    """
    Calls a function on an array ignoring np.nan and +/- np.inf.
    """
    ma = np.isfinite(arr)
    return f(arr[ma])


def OP_constants(xs, degree=3, tol=10**-14):
    """
    For constructing orthagonal polynomial functions of the general form:
    y(x) = a_0 + a_1 * (x - β) + a_2 * (x - γ_0) * (x - γ_1) + \
           a_3 * (x - δ_0) * (x - δ_1) * (x - δ_2)
    Finds the parameters (β_0), (γ_0, γ_1), (δ_0, δ_1, δ_2).

    These parameters are functions only of the independent variable x.
    """
    xs = np.array(xs)
    x = var('x')
    params = []
    for d in range(degree):
        ps = symbols('{}0:{}'.format(chr(945+d),d))
        logger.debug('Generating {} DIM {} equations for {}.'.format(d, d, ps))
        if d:
            eqs = []
            for _deg in range(d):
                q = 1
                if _deg:
                    q = x ** _deg
                for p in ps:
                    q *= (x - p)
                eqs.append(q)

            sums = []
            for q in eqs:
                sumq = 0.
                for xi in xs:
                    sumq += q.subs(dict(x=xi))
                sums.append(sumq)

            guess = np.linspace(np.nanmin(xs), np.nanmax(xs), d+2)[1:-1]
            result = nsolve(sums, ps, list(guess), tol=tol)
            params.append(tuple(result))
        else:
            params.append(()) # first parameter
    return params


def lambda_poly(x, ps):
    """
    Polynomial lambda_n(x) given parameters ps with len(ps) = n.

    Parameters:
    -----------
    x: np.ndarray
        X values to calculate the function at.
    ps: tuple
        Parameter set tuple. E.g. parameters (a, b) from f(x) = (x-a)(x-b).
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    result = np.ones(len(x))
    for p in ps:
        result = result * (x - p)
    return result.astype(np.float)


def lambda_min_func(ls, ys, arrs, power=1.):
    cost = np.abs(np.dot(ls, arrs) - ys)**power
    cost[np.isnan(cost)] = 0.
    return cost


def lambdas(arr:np.ndarray,
            xs=np.array([]),
            params=None,
            degree=5,
            costf_power=1.,
            residuals=False,
            min_func=lambda_min_func):
    """
    Parameterises values based on linear combination of orthagonal polynomials
    over a given set of x values.
    """
    if params is None:
        params = OP_constants(xs, degree=degree)

    fs = np.array([lambda_poly(xs, pset) for pset in params])

    guess = np.zeros(degree)
    result = optimize.least_squares(min_func,
                                    guess,
                                    args=(arr, fs, costf_power)) # , method='Nelder-Mead'
    if residuals:
        return result.x, result.fun
    else:
        return result.x


def lambda_poly_func(lambdas:np.ndarray,
                     params=None,
                     pxs=None,
                     degree=5):
    """
    Expansion of lambda parameters back to the original space. Returns a
    function which evaluates the sum of the orthaogonal polynomials at given
    x values.

    Parameters:
    ------------
    lambdas: np.ndarray
        Lambda values to weight combination of polynomials.
    params: list of tuples
        Parameters for the orthagonal polynomial decomposition.
    pxs: np.ndarray
        x values used to construct the lambda values.*
    degree: int
        Degree of the orthagonal polynomial decomposition.*

    * (only needed if parameters are not supplied)
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

        Parameters:
        -----------
        xarr: np.ndarray
            X values at which to evaluate the function.
        """
        arrs = np.array([lambda_poly(xarr, pset) for pset in params])
        return np.dot(lambdas, arrs)

    return lambda_poly_f
