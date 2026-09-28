from typing import Callable
from copy import copy

import numpy as np
import scipy
import sympy

from .log import Handle

logger = Handle(__name__)


def eigsorted(cov: np.ndarray[tuple[int, int], np.dtype[np.floating]]):
    """
    Returns arrays of eigenvalues and eigenvectors sorted by magnitude.

    Parameters
    -----------
    cov : numpy.ndarray
        Covariance matrix to extract eigenvalues and eigenvectors from.

    Returns
    --------
    vals : numpy.ndarray
        Sorted eigenvalues.
    vecs : numpy.ndarray
        Sorted eigenvectors.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def augmented_covariance_matrix(
    M: np.ndarray[tuple[int, int], np.dtype[np.number]],
    C: np.ndarray[tuple[int, int], np.dtype[np.floating]],
):
    r"""
    Constructs an augmented covariance matrix from means M and covariance matrix C.

    Parameters
    ----------
    M : numpy.ndarray
        Array of means.
    C : numpy.ndarray
        Covariance matrix.

    Returns
    ---------
    numpy.ndarray
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


def interpolate_line(
    x: np.ndarray[tuple[int]], y: np.ndarray[tuple[int]], n: int = 0, logy: bool = False
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.floating]],
    np.ndarray[tuple[int], np.dtype[np.floating]],
]:
    """
    Add intermediate evenly spaced points interpolated between given x-y coordinates,
    assuming the x points are the same.

    Parameters
    -----------
    x : numpy.ndarray
        1D array of x values.

    y : numpy.ndarray
        ND array of y values.
    """
    if logy:  # perform interpolation against logy, then revert with exp
        y = np.log(y)

    current = x[:-1].copy()  # the first part of the x array
    intervals = x[1:] - x[:-1]  # right-wise intervals (could be negative for REE)
    _x = current.copy().astype(float)

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


def grid_from_ranges(
    X: np.ndarray, bins: int | list[int] = 100, **kwargs
) -> tuple[np.ndarray, ...]:
    """
    Create a meshgrid based on the ranges along columns of array X.

    Parameters
    -----------
    X : numpy.ndarray
        Array of shape `(samples, dimensions)` to create a meshgrid from.
    bins : int | tuple
        Shape of the meshgrid. If an integer, provides a square mesh. If a tuple,
        values for each column are required.

    Returns
    --------
    numpy.ndarray

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


def flattengrid(grid: tuple[np.ndarray, ...]) -> np.ndarray[tuple[int, int]]:
    """
    Convert a collection of arrays to a concatenated array of flattened components.
    Useful for passing meshgrid values to a function which accepts argumnets of shape
    `(samples, dimensions)`.

    Parameters
    -----------
    grid : list
        Collection of arrays (e.g. a meshgrid) to flatten and concatenate.

    Returns
    --------
    numpy.ndarray
    """
    return np.vstack([g.flatten() for g in grid]).T


def linspc_(
    _min: float, _max: float, step: float = 0.0, bins: int = 20
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """
    Linear spaced array, with optional step for grid margins.

    Parameters
    -----------
    _min : float
        Minimum value for spaced range.
    _max : float
        Maximum value for spaced range.
    step : float, 0.0
        Step for expanding at grid edges. Default of 0.0 results in no expansion.
    bins : int
        Number of bins to divide the range (adds one by default).

    Returns
    -------
    numpy.ndarray
        Linearly-spaced array.
    """
    if step < 0:
        step = -step
    return np.linspace(_min - step, _max + step, bins + 1)


def logspc_(
    _min: float, _max: float, step: float = 1.0, bins: int = 20
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """
    Log spaced array, with optional step for grid margins.

    Parameters
    -----------
    _min : float
        Minimum value for spaced range.
    _max : float
        Maximum value for spaced range.
    step : float, 1.0
        Step for expanding at grid edges. Default of 1.0 results in no expansion.
    bins : int
        Number of bins to divide the range (adds one by default).

    Returns
    -------
    numpy.ndarray
        Log-spaced array.
    """
    if step < 1.0:
        step = 1.0 / step
    return np.logspace(np.log(_min / step), np.log(_max * step), bins, base=np.e)


def logrng_(v: list[float] | np.ndarray, exp: float = 0.0) -> tuple[float, float]:
    """
    Range of a sample, where values <0 are excluded.

    Parameters
    -----------
    v : list; list-like
        Array of values to obtain a range from.
    exp : float, (0, 1)
        Fractional expansion of the range.

    Returns
    -------
    tuple
        Min, max tuple.
    """
    v = np.array(v)
    u = v[(v > 0)]  # make sure the range_values are >0
    return linrng_(u, exp=exp)


def linrng_(v: list[float] | np.ndarray, exp: float = 0.0) -> tuple[float, float]:
    """
    Range of a sample, where values <0 are included.

    Parameters
    -----------
    v : list; list-like
        Array of values to obtain a range from.
    exp : float, (0, 1)
        Fractional expansion of the range.

    Returns
    -------
    tuple
        Min, max tuple.
    """
    v = np.array(v)
    u = v[np.isfinite(v)]
    return (np.nanmin(u) * (1.0 - exp), np.nanmax(u) * (1.0 + exp))


def isclose(
    a: float | np.ndarray, b: float | np.ndarray
) -> bool | np.ndarray[tuple[int, ...], np.dtype[np.bool]]:
    """
    Implementation of np.isclose with equal nan.


    Parameters
    ------------
    a,b : float | numpy.ndarray
        Numbers or arrays to compare.
    Returns
    -------
    bool
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


def is_numeric(obj) -> bool:
    """
    Check for numerical behaviour.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    --------
    bool
    """

    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


@np.vectorize
def round_sig(x: float | np.ndarray, sig: int = 2) -> float | np.ndarray:
    """
    Round a number to a certain number of significant figures.

    Parameters
    ----------
    x : float
        Number to round.
    sig : int
        Number of significant digits to round to.

    Returns
    -------
    float
    """
    where_nan = ~np.isfinite(x)
    x = copy(x)
    if hasattr(x, "__len__"):
        x[where_nan] = np.finfo(np.float64).eps
        vals = np.round(x, sig - int(np.floor(np.log10(np.abs(x)))) - 1)
        vals[where_nan] = np.nan
        return vals
    else:
        try:
            return np.round(x, sig - int(np.floor(np.log10(np.abs(x)))) - 1)
        except (ValueError, OverflowError):  # nan or inf is passed
            return x


def significant_figures(
    n: float | np.ndarray,
    unc: float | np.ndarray | None = None,
    max_sf: int = 20,
    rtol: float = 1e-20,
) -> int | np.ndarray:
    """
    Iterative method to determine the number of significant digits for a given float,
    optionally providing an uncertainty.

    Parameters
    ----------
    n : float
        Number from which to ascertain the significance level.
    unc : float, `None`
        Uncertainty, which if provided is used to derive the number of significant
        digits.
    max_sf : int
        An upper limit to the number of significant digits suggested.
    rtol : float
        Relative tolerance to determine similarity of numbers, used in calculations.

    Returns
    -------
    int
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
                np.vstack([np.zeros(mag_n.shape), (1.0 + mag_n - mag_u).astype(int)]),
                axis=0,
            ).astype(int)
        else:
            rounded = np.vstack([_n] * max_sf).reshape(max_sf, *_n.shape)
            indx = np.indices(rounded.shape)[0]  # get the row indexes for no. sig figs
            rounded = round_sig(rounded, indx)
            sfs = np.nanargmax(np.isclose(rounded, _n, rtol=rtol), axis=0)
        sfs[np.isnan(sfs)] = 0
        return sfs


def signify_digit(
    n: float | np.ndarray,
    unc: float | np.ndarray | None = None,
    leeway: int = 0,
    low_filter: bool = True,
):
    """
    Reformats numbers to contain only significant_digits. Uncertainty can be provided to
    digits with relevant precision.

    Parameters
    ----------
    n : float
        Number to reformat
    unc : float, `None`
        Absolute uncertainty on the number, optional.
    leeway : int, 0
        Manual override for significant figures. Positive values will force extra
        significant figures; negative values will remove significant figures.
    low_filter : bool, `True`
        Whether to return `np.nan` in place of values which are within precision
        equal to zero.

    Returns
    -------
    float
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

                def fmt(x):
                    return x

            sig_n = round(n, round_to)
            if low_filter and sig_n == 0.0:
                return np.nan
            else:
                return fmt(sig_n)
    else:
        return np.nan


def most_precise(arr: np.ndarray) -> float | np.ndarray:
    """
    Get the most precise element from an array.

    Parameters
    -----------
    arr : numpy.ndarray
        Array to obtain the most precise element/subarray from.

    Returns
    -----------
    float | numpy.ndarray
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


def equal_within_significance(
    arr: np.ndarray, equal_nan: bool = False, rtol: float = 1e-15
) -> bool | np.ndarray:
    """
    Test whether elements of an array are equal within the precision of the
    least precise.

    Parameters
    ------------
    arr : numpy.ndarray
        Array to test.
    equal_nan : bool, `False`
        Whether to consider `np.nan` elements equal to one another.
    rtol : float
        Relative tolerance for comparison.

    Returns
    ---------
    bool | numpy.ndarray(bool)
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


def helmert_basis(D: int, full: bool = False, **kwargs) -> np.ndarray[tuple[int, int]]:
    """
    Generate a set of orthogonal basis vectors in the form of a helmert matrix.

    Parameters
    ---------------
    D : int
        Dimension of compositional vectors.

    Returns
    --------
    numpy.ndarray
        (D-1, D) helmert matrix corresponding to default orthogonal basis.
    """
    H = scipy.linalg.helmert(D, full=full, **kwargs)
    return H


def symbolic_helmert_basis(
    D: int, full: bool = False
) -> sympy.matrices.dense.DenseMatrix:
    """
    Get a symbolic representation of a Helmert Matrix.

    Parameters
    ----------
    D : int
        Order of the matrix. Equivalent to dimensionality for compositional data
        analysis.
    full : bool
        Whether to return the full matrix, or alternatively exclude the first row.
        Analogous to the option for `scipy.linalg.helmert`.

    Returns
    --------
    sympy.matrices.dense.DenseMatrix
    """

    rows = []
    if full:
        rows += [[1 / sympy.sqrt(D)] * D]

    for r in np.arange(1, D):
        rows += [
            [1 / sympy.sqrt((r + 1) * r)] * r  # 1/sqrt(n(*n+1))
            + [-r / sympy.sqrt((r + 1) * r)]  # -n/sqrt(n(*n+1))
            + [0] * int(D - r - 1)
        ]
    # could check summations here

    return sympy.Matrix(rows)


def on_finite(X: np.ndarray, f: Callable) -> np.ndarray:
    """
    Calls a function on an array ignoring np.nan and +/- np.inf. Note that the
    shape of the output may be different to that of the input.

    Parameters
    ---------------
    X : numpy.ndarray
        Array on which to perform the function.
    f : Callable
        Function to call on the array.

    Returns
    -------
    numpy.ndarray
    """
    ma = np.isfinite(X)
    return f(X[ma])


def nancov(X: np.ndarray[tuple[int, int]]) -> np.ndarray[tuple[int, int]]:
    """
    Generates a covariance matrix excluding nan-components.

    Parameters
    ---------------
    X : numpy.ndarray
        Input array for which to derive a covariance matrix.

    Returns
    -------
    numpy.ndarray
    """
    # tried and true - simply excludes samples
    Xnanfree = X[np.all(np.isfinite(X), axis=1), :].T
    # assert Xnanfree.shape[1] > Xnanfree.shape[0]
    # (1/m)X^T*X
    return np.cov(Xnanfree)


def solve_ratios(*eqs, evaluate: bool = True) -> list[float]:
    """
    Solve a ternary system (top-left-right) given two constraints on
    two ratios, which together describe intersecting lines/a point.
    """
    t, L, r = sympy.symbols("t l r")

    def to_sympy(t):  # rearrange to have =0 equvalent expressions
        return sympy.sympify("-".join(t.split("=")))

    result = sympy.solve(
        [to_sympy(e) for e in eqs] + [to_sympy("t + l + r = 1")], (t, L, r)
    )
    return list(result.values())
