import numpy as np
from sympy.solvers.solvers import nsolve
from sympy import symbols, var
from functools import partial
import scipy
import logging
from copy import copy
from .meta import update_docstring_references


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def eigsorted(cov):
    """
    Returns arrays of eigenvalues and eigenvectors sorted by magnitude.

    Parameters
    -----------
    cov : :class:`numpy.ndarray`
        Covariance matrix to extract eigenvalues and eigenvectors from.

    Returns
    --------
    vals : :class:`numpy.ndarray`
        Sorted eigenvalues.
    vecs : :class:`numpy.ndarray`
        Sorted eigenvectors.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def augmented_covariance_matrix(M, C):
    r"""
    Constructs an augmented covariance matrix from means M and covariance matrix C.

    Parameters
    ----------
    M : :class:`numpy.ndarray`
        Array of means.
    C : :class:`numpy.ndarray`
        Covariance matrix.

    Returns
    ---------
    :class:`numpy.ndarray`
        Augmented covariance matrix A.

    Notes
    ------
        Augmented covariance matrix constructed from mean of shape (D, ) and covariance
        matrix of shape (D, D) as follows:

        .. math::
                \begin{array}{c|c}
                -1 & M.T \\
                \hline
                M & C
                \end{array}
    """
    d = np.squeeze(M).shape[0]
    A = np.zeros((d + 1, d + 1))
    A[0, 0] = -1
    A[0, 1 : d + 1] = M
    A[1 : d + 1, 0] = M.T
    A[1 : d + 1, 1 : d + 1] = C
    return A


def interpolate_line(xy, n=0):
    """
    Add intermediate evenly spaced points interpolated between given x-y coordinates.
    """
    xy = np.squeeze(xy)
    if xy.ndim > 2:
        return np.array(list(map(lambda line: interpolate_line(line, n=n), xy)))
    x, y = xy
    intervals = x[1:] - x[:-1]
    current = x[:-1].copy()
    _x = current.copy().astype(np.float)
    if n:
        dx = intervals / (n + 1.0)
        for ix in range(n):
            current = current + dx
            _x = np.append(_x, current)
    _x = np.append(_x, np.array([x[-1]]))
    _x = np.sort(_x)
    _y = np.interp(_x, x, y)
    assert all([i in _x for i in x])
    return np.vstack([_x, _y])


def grid_from_ranges(X, bins=100, **kwargs):
    """
    Create a meshgrid based on the ranges along columns of array X.

    Parameters
    -----------
    X : :class:`numpy.ndarray`
        Array of shape :code:`(samples, dimensions)` to create a meshgrid from.
    bins : :class:`int` | :class:`tuple`
        Shape of the meshgrid. If an integer, provides a square mesh. If a tuple,
        values for each column are required.

    Returns
    --------
    :class:`numpy.ndarray`

    Notes
    -------
    Can pass keyword arg indexing = {'xy', 'ij'}
    """
    dim = X.shape[1]
    if isinstance(bins, int):  # expand to list of len == dimensions
        bins = [bins for ix in range(dim)]
    mmb = [(np.nanmin(X[:, ix]), np.nanmax(X[:, ix]), bins[ix]) for ix in range(dim)]
    grid = np.meshgrid(*[np.linspace(*i) for i in mmb], **kwargs)
    return grid


def flattengrid(grid):
    """
    Convert a collection of arrays to a concatenated array of flattened components.
    Useful for passing meshgrid values to a function which accepts argumnets of shape
    :code:`(samples, dimensions)`.

    Parameters
    -----------
    grid : :class:`list`
        Collection of arrays (e.g. a meshgrid) to flatten and concatenate.


    Returns
    --------
    :class:`numpy.ndarray`
    """
    return np.c_[[g.ravel() for g in grid]].T


def linspc_(_min, _max, step=0.0, bins=20):
    """
    Linear spaced array, with optional step for grid margins.

    Parameters
    -----------
    _min : :class:`float`
        Minimum value for spaced range.
    _max : :class:`float`
        Maximum value for spaced range.
    step : :class:`float`, 0.0
        Step for expanding at grid edges. Default of 0.0 results in no expansion.
    bins : int
        Number of bins to divide the range (adds one by default).

    Returns
    -------
    :class:`numpy.ndarray`
        Linearly-spaced array.
    """
    return np.linspace(_min - step, _max + step, bins + 1)


def logspc_(_min, _max, step=1.0, bins=20):
    """
    Log spaced array, with optional step for grid margins.

    Parameters
    -----------
    _min : :class:`float`
        Minimum value for spaced range.
    _max : :class:`float`
        Maximum value for spaced range.
    step : :class:`float`, 1.0
        Step for expanding at grid edges. Default of 1.0 results in no expansion.
    bins : int
        Number of bins to divide the range (adds one by default).

    Returns
    -------
    :class:`numpy.ndarray`
        Log-spaced array.
    """
    return np.logspace(np.log(_min / step), np.log(_max * step), bins, base=np.e)


def logrng_(v, exp=0.0):
    """
    Range of a sample, where values <0 are excluded.

    Parameters
    -----------
    v : :class:`list`; list-like
        Array of values to obtain a range from.
    exp : :class:`float`, (0, 1)
        Fractional expansion of the range.

    Returns
    -------
    :class:`tuple`
        Min, max tuple.
    """
    u = v[(v > 0)]  # make sure the range_values are >0
    return linrng_(u, exp=exp)


def linrng_(v, exp=0.0):
    """
    Range of a sample, where values <0 are included.

    Parameters
    -----------
    v : :class:`list`; list-like
        Array of values to obtain a range from.
    exp : :class:`float`, (0, 1)
        Fractional expansion of the range.

    Returns
    -------
    :class:`tuple`
        Min, max tuple.
    """
    u = v[np.isfinite(v)]
    return (np.nanmin(u) * (1.0 - exp), np.nanmax(u) * (1.0 + exp))


def isclose(a, b):
    """
    Implementation of np.isclose with equal nan.


    Parameters
    ------------
    a,b : :class:`float` | :class:`numpy.ndarray`
        Numbers or arrays to compare.
    Returns
    -------
    :class:`bool`
    """
    hasnan = np.isnan(a) | np.isnan(b)
    if np.array(a).ndim > 1:
        if hasnan.any():
            # if they're both all nan in the same places
            if not np.isnan(a[hasnan]).all() & np.isnan(b[hasnan]).all():
                return False
            else:
                return np.isclose(a[~hasnan], b[~hasnan])
        else:
            return np.isclose(a, b)
    else:
        if hasnan:
            return np.isnan(a) & np.isnan(b)
        else:
            return np.isclose(a, b)


def is_numeric(obj):
    """
    Check for numerical behaviour.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    --------
    :class:`bool`
    """

    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


@np.vectorize
def round_sig(x, sig=2):
    """
    Round a number to a certain number of significant figures.

    Parameters
    ----------
    x : :class:`float`
        Number to round.
    sig : :class:`int`
        Number of significant digits to round to.

    Returns
    -------
    :class:`float`
    """
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
    """
    Iterative method to determine the number of significant digits for a given float,
    optionally providing an uncertainty.

    Parameters
    ----------
    n : :class:`float`
        Number from which to ascertain the significance level.
    unc : :class:`float`, :code:`None`
        Uncertainty, which if provided is used to derive the number of significant
        digits.
    max_sf : :class:`int`
        An upper limit to the number of significant digits suggested.
    rtol : :class:`float`
        Relative tolerance to determine similarity of numbers, used in calculations.

    Returns
    -------
    :class:`int`
        Number of significant digits.
    """
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
    """
    Get the most precise element from an array.

    Parameters
    -----------
    arr : :class:`numpy.ndarray`
        Array to obtain the most precise element/subarray from.

    Returns
    -----------
    :class:`float` | :class:`numpy.ndarray`
        Returns the most precise array element (for ndim=1), or most precise subarray
        (for ndim > 1).
    """
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
    Test whether elements of an array are equal within the precision of the
    least precise.

    Parameters
    ------------
    arr : :class:`numpy.ndarray`
        Array to test.
    equal_nan : :class:`bool`, :code:`False`
        Whether to consider :class:`np.nan` elements equal to one another.
    rtol : :class:`float`
        Relative tolerance for comparison.

    Returns
    ---------
    :class:`bool` | :class:`numpy.ndarray`(:class:`bool`)
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

    Parameters
    ----------
    n : :class:`float`
        Number to reformat
    unc : :class:`float`, :code:`None`
        Absolute uncertainty on the number, optional.
    leeway : :class:`int`, 0
        Manual override for significant figures. Positive values will force extra
        significant figures; negative values will remove significant figures.
    low_filter : :class:`bool`, :code:`True`
        Whether to return :class:`np.nan` in place of values which are within precision
        equal to zero.

    Returns
    -------
    :class:`float`
        Reformatted number.

    Notes
    -----
        * Will not pad 0s at the end or before floats.
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


def orthogonal_basis_default(D: int, **kwargs):
    """
    Generate a set of orthogonal basis vectors .

    Parameters
    ---------------
    D : :class:`int`
        Dimension of compositional vectors.

    Returns
    --------
    :class:`numpy.ndarray`
        (D-1, D) helmert matrix corresponding to default orthogonal basis.
    """
    H = scipy.linalg.helmert(D, **kwargs)
    return H[::-1]


def orthogonal_basis_from_array(X: np.ndarray, **kwargs):
    """
    Generate a set of orthogonal basis vectors.

    Parameters
    ---------------
    X : :class:`numpy.ndarray`
        Array from which the size of the set is derived.

    Returns
    --------
    :class:`numpy.ndarray`
        (D-1, D) helmert matrix corresponding to default orthogonal basis.

    Notes
    -----
        * Currently returns the default set of basis vectors for an array of given dim.

    Todo
    -----
        * Update to provide other potential sets of basis vectors.
    """
    return orthogonal_basis_default(X.shape[1], **kwargs)


def on_finite(X, f):
    """
    Calls a function on an array ignoring np.nan and +/- np.inf. Note that the
    shape of the output may be different to that of the input.

    Parameters
    ---------------
    X : :class:`numpy.ndarray`
        Array on which to perform the function.
    f : :class:`Callable`
        Function to call on the array.

    Returns
    -------
    :class:`numpy.ndarray`
    """
    ma = np.isfinite(X)
    return f(X[ma])


def nancov(X):
    """
    Generates a covariance matrix excluding nan-components.

    Parameters
    ---------------
    X : :class:`numpy.ndarray`
        Input array for which to derive a covariance matrix.

    Returns
    -------
    :class:`numpy.ndarray`
    """
    # tried and true - simply excludes samples
    Xnanfree = X[np.all(np.isfinite(X), axis=1), :].T
    # assert Xnanfree.shape[1] > Xnanfree.shape[0]
    # (1/m)X^T*X
    return np.cov(Xnanfree)


@update_docstring_references
def OP_constants(xs, degree=3, tol=10 ** -14):
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
        Target data to fit.
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
