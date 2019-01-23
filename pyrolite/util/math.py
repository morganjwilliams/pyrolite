import numpy as np
from sympy.solvers.solvers import nsolve
from sympy import symbols, var
from functools import partial
import scipy
import logging
from copy import copy


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def is_numeric(obj):
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


@np.vectorize
def round_sig(x, sig=2):
    """Round a number to a certain number of significant figures."""
    where_nan = ~np.isfinite(x)
    x = copy(x)
    if hasattr(x, "__len__"):
        x[where_nan] = np.finfo(np.float).eps
        vals = np.round(x, sig - np.int(np.floor(np.log10(np.abs(x)))) - 1)
        vals[where_nan] = np.nan
        return vals
    else:
        try:
            return np.round(x, sig - np.int(np.floor(np.log10(np.abs(x)))) - 1)
        except (ValueError, OverflowError):  # nan or inf is passed
            return x


def significant_figures(n, unc=None, max_sf=20, rtol=1e-20):
    """Get number of significant digits for a number, given an uncertainty."""
    if not hasattr(n, "__len__"):
        if np.isfinite(n):
            if unc is not None:
                mag_n = np.floor(np.log10(np.abs(n)))
                mag_u = np.floor(np.log10(unc))
                if not np.isfinite(mag_u) or not np.isfinite(mag_n):
                    return np.nan
                sf = int(max(0, int(1.0 + mag_n - mag_u)))
            else:
                sf = min(
                    [
                        ix
                        for ix in range(max_sf)
                        if np.isclose(round_sig(n, ix), n, rtol=rtol)
                    ]
                )
            return sf
        else:
            return 0
    else:  # this isn't working
        n = np.array(n)
        _n = n.copy()
        mask = np.isclose(n, 0.0)  # can't process zeros
        _n[mask] = np.nan
        if unc is not None:
            mag_n = np.floor(np.log10(np.abs(_n)))
            mag_u = np.floor(np.log10(unc))
            sfs = np.nanmax(
                np.vstack(
                    [np.zeros(mag_n.shape), (1.0 + mag_n - mag_u).astype(np.int)]
                ),
                axis=0,
            ).astype(np.int)
        else:
            rounded = np.vstack([_n] * max_sf).reshape(max_sf, *_n.shape)
            indx = np.indices(rounded.shape)[0]  # get the row indexes for no. sig figs
            rounded = round_sig(rounded, indx)
            sfs = np.nanargmax(np.isclose(rounded, _n, rtol=rtol), axis=0)
        sfs[np.isnan(sfs)] = 0
        return sfs


def most_precise(arr):
    """Get the most precise element from an array."""
    arr = np.array(arr)
    if np.isfinite(arr).any().any():
        precision = significant_figures(arr)
        if arr.ndim > 1:
            return arr[range(arr.shape[0]), np.nanargmax(precision, axis=-1)]
        else:
            return arr[np.nanargmax(precision, axis=-1)]
    else:
        return np.nan


def equal_within_significance(arr, equal_nan=False, rtol=1e-15):
    """
    Test whether elements within an array are equal to the precision of the
    least precise.
    """
    arr = np.array(arr)

    if arr.ndim == 1:
        if not np.isfinite(arr).all():
            return equal_nan
        else:
            precision = significant_figures(arr)
            min_precision = np.nanmin(precision)
            rounded = round_sig(arr, min_precision * np.ones(arr.shape, dtype=int))
            return np.isclose(rounded[0], rounded, rtol=rtol).all()
    else:  # ndmim =2
        equal = equal_nan * np.ones(
            arr.shape[0], dtype=bool
        )  # mean for rows containing nan
        if np.isfinite(arr).all(axis=1).any():
            non_nan_rows = np.isfinite(arr).all(axis=1)

            precision = significant_figures(arr[non_nan_rows, :])
            min_precision = np.nanmin(precision, axis=1)
            precs = np.repeat(min_precision, arr.shape[1]).reshape(
                arr[non_nan_rows, :].shape
            )
            rounded = round_sig(arr[non_nan_rows, :], precs)
            equal[non_nan_rows] = np.apply_along_axis(
                lambda x: (x == x[0]).all(), 1, rounded
            )

        return equal


def signify_digit(n, unc=None, leeway=0, low_filter=True):
    """
    Reformats numbers to contain only significant_digits. Uncertainty can be provided to
    digits with relevant precision.

    Note: Will not pad 0s at the end or before floats.
    """

    if np.isfinite(n):
        if np.isclose(n, 0.0):
            return n
        else:
            mag_n = np.floor(np.log10(np.abs(n)))
            sf = significant_figures(n, unc=unc) + int(leeway)
            if unc is not None:
                mag_u = np.floor(np.log10(unc))
            else:
                mag_u = 0
            round_to = sf - int(mag_n) - 1 + leeway
            if round_to <= 0:
                fmt = int
            else:
                fmt = lambda x: x
            sig_n = round(n, round_to)
            if low_filter and sig_n == 0.0:
                return np.nan
            else:
                return fmt(sig_n)
    else:
        return np.nan


def orthagonal_basis(X: np.ndarray):
    """
    Generate a set of orthagonal basis vectors.

    Parameters
    ---------------
    X : np.ndarray
        Array from which the size of the set is derived.
    """
    D = X.shape[1]
    # D-1, D Helmert matrix, exact representation of ψ as in Egozogue's book
    H = scipy.linalg.helmert(D, full=False)
    return H[::-1]


def on_finite(X, f):
    """
    Calls a function on an array ignoring np.nan and +/- np.inf. Note that the
    shape of the output may be different to that of the input.

    Parameters
    ---------------
    X : np.ndarray
        Array on which to perform the function.
    f : function
        Function to call on the array.
    """
    ma = np.isfinite(X)
    return f(X[ma])


def nancov(X, method="replace"):
    """
    Generates a covariance matrix excluding nan-components.  Done on a
    column-column/pairwise basis. The result Y may not be a positive definite matrix.

    Parameters
    ---------------
    X: np.ndarray
        Input array for which to derive a covariance matrix.
    method: str, 'row_exclude' | 'replace'
        Method for calculating covariance matrix.
        'row_exclude' removes all rows  which contain np.nan before calculating
        the covariance matrix. 'replace' instead replaces the np.nan values with
        the mean before calculating the covariance.

    """
    if method == "rowexclude":
        Xnanfree = X[np.all(np.isfinite(X), axis=1), :].T
        # assert Xnanfree.shape[1] > Xnanfree.shape[0]
        # (1/m)X^T*X
        return np.cov(Xnanfree)
    else:
        X = np.array(X, ndmin=2, dtype=float)
        X -= np.nanmean(X, axis=0)  # [:, np.newaxis]
        cov = np.empty((X.shape[1], X.shape[1]))
        cols = range(X.shape[1])
        for n in cols:
            for m in [i for i in cols if i >= n]:
                fn = np.isfinite(X[:, n])
                fm = np.isfinite(X[:, m])
                if method == "replace":
                    X[~fn, n] = 0
                    X[~fm, m] = 0
                    fact = fn.shape[0] - 1
                    c = np.dot(X[:, n], X[:, m]) / fact
                else:
                    f = fn & fm
                    fact = f.shape[0] - 1
                    c = np.dot(X[f, n], X[f, m]) / fact
                cov[n, m] = c
                cov[m, n] = c
        return cov


def OP_constants(xs, degree=3, tol=10 ** -14):
    """
    For constructing orthagonal polynomial functions of the general form:
    y(x) = a_0 + a_1 * (x - β) + a_2 * (x - γ_0) * (x - γ_1) +
    a_3 * (x - δ_0) * (x - δ_1) * (x - δ_2)
    Finds the parameters (β_0), (γ_0, γ_1), (δ_0, δ_1, δ_2).

    These parameters are functions only of the independent variable x.
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
            params.append(tuple(result))
        else:
            params.append(())  # first parameter
    return params


def lambda_poly(x, ps):
    """
    Polynomial lambda_n(x) given parameters ps with len(ps) = n.

    Parameters
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


def lambda_min_func(ls, ys, arrs, power=2.0):
    cost = np.abs(np.dot(ls, arrs) - ys) ** power
    cost[np.isnan(cost)] = 0.0  # can't change nans - don't penalise them
    return cost


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
    Parameterises values based on linear combination of orthagonal polynomials
    over a given set of x values.
    """
    if np.isnan(arr).any():  # With missing data, the method can't be used.
        x = np.nan * np.ones(degree)
        res = np.nan * np.ones(degree)
    else:
        guess = guess or np.exp(np.arange(degree) + 2)
        params = params or OP_constants(xs, degree=degree)

        fs = np.array([lambda_poly(xs, pset) for pset in params])

        result = scipy.optimize.least_squares(
            min_func, guess, args=(arr, fs, costf_power)  # , method='Nelder-Mead'
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
    function which evaluates the sum of the orthaogonal polynomials at given
    x values.

    Parameters
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

        Parameters
        -----------
        xarr: np.ndarray
            X values at which to evaluate the function.
        """
        arrs = np.array([lambda_poly(xarr, pset) for pset in params])
        return np.dot(lambdas, arrs)

    return lambda_poly_f
