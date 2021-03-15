import numpy as np
import sympy
import scipy
from copy import copy
from .meta import update_docstring_references
from .log import Handle

logger = Handle(__name__)


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


def interpolate_line(x, y, n=0, logy=False):
    """
    Add intermediate evenly spaced points interpolated between given x-y coordinates,
    assuming the x points are the same.

    Parameters
    -----------
    x : :class:`numpy.ndarray`
        1D array of x values.

    y : :class:`numpy.ndarray`
        ND array of y values.
    """
    if logy:  # perform interpolation against logy, then revert with exp
        y = np.log(y)

    current = x[:-1].copy() # the first part of the x array
    intervals = x[1:] - x[:-1] # right-wise intervals (could be negative for REE)
    _x = current.copy().astype(np.float)

    if n:  # should be able to tile this instead
        dx = intervals / (n + 1.0)
        for ix in range(n):
            current = current + dx
            _x = np.hstack([_x, current])

    _x = np.append(_x, x[-1])  # add one final value to x series
    _x = np.sort(_x, axis=-1)
    f = scipy.interpolate.interp1d(x, y, axis=-1)
    _y = f(_x)
    # assert all([i in _x for i in x])
    if logy:
        _y = np.exp(_y)
    return _x, _y


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
    return np.vstack([g.flatten() for g in grid]).T


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
    if step < 0:
        step = -step
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
    if step < 1.0:
        step = 1.0 / step
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
                if np.isnan(mag_u):
                    mag_u = 0
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


def helmert_basis(D: int, full=False, **kwargs):
    """
    Generate a set of orthogonal basis vectors in the form of a helmert matrix.

    Parameters
    ---------------
    D : :class:`int`
        Dimension of compositional vectors.

    Returns
    --------
    :class:`numpy.ndarray`
        (D-1, D) helmert matrix corresponding to default orthogonal basis.
    """
    H = scipy.linalg.helmert(D, full=full, **kwargs)
    return H


def symbolic_helmert_basis(D, full=False):
    """
    Get a symbolic representation of a Helmert Matrix.

    Parameters
    ----------
    D : :class:`int`
        Order of the matrix. Equivalent to dimensionality for compositional data
        analysis.
    full : :class:`bool`
        Whether to return the full matrix, or alternatively exclude the first row.
        Analogous to the option for :func:`scipy.linalg.helmert`.

    Returns
    --------
    :class:`sympy.matrices.dense.DenseMatrix`
    """

    rows = []
    if full:
        rows += [[1 / sympy.sqrt(D)] * D]

    for r in np.arange(1, D):
        rows += [
            [1 / sympy.sqrt((r + 1) * r)] * r  # 1/sqrt(n(*n+1))
            + [-r / sympy.sqrt((r + 1) * r)]  # -n/sqrt(n(*n+1))
            + [0] * (D - r - 1)
        ]
    # could check summations here

    return sympy.Matrix(rows)


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
