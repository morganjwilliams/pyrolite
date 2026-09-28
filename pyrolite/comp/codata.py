from traitlets import Float
from periodictable import Ca
import warnings
from collections.abc import Callable
from typing import overload

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import sympy

from ..util.log import Handle

# from .renorm import renormalise, close
from ..util.math import helmert_basis, symbolic_helmert_basis

logger = Handle(__name__)

__TRANSFORMS__: dict[str, tuple[Callable, Callable]] = {}

__sympy_protected_variables__: dict[str, str] = {"S": "Ss"}


@overload
def close(
    X: np.ndarray[tuple[int, int], np.dtype[np.number]], sumf: Callable
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]: ...
@overload
def close(
    X: np.ndarray[tuple[int], np.dtype[np.number]], sumf: Callable
) -> np.ndarray[tuple[int], np.dtype[np.floating]]: ...
def close(
    X: np.ndarray[tuple[int, int] | tuple[int], np.dtype[np.number]],
    sumf: Callable = np.sum,
) -> np.ndarray[tuple[int, int] | tuple[int], np.dtype[np.floating]]:
    """
    Closure operator for compositional data.

    Parameters
    -----------
    X : numpy.ndarray
        Array to close.
    sumf : Callable, `numpy.sum`
        Sum function to use for closure.

    Returns
    --------
    numpy.ndarray
        Closed array.

    Notes
    ------
    Checks for non-positive entries and replaces zeros with NaN values.
    """

    if np.any(X <= 0):
        warnings.warn(
            "Non-positive entries found. Closure operation assumes all positive entries.",
            UserWarning,
        )

    if X.ndim == 2:
        C: np.ndarray[tuple[int, int], np.dtype[np.floating]] = np.array(
            sumf(X, axis=1), dtype=float
        )[:, np.newaxis]
    else:
        C: np.ndarray[tuple[int], np.dtype[np.floating]] = np.array(
            sumf(X), dtype=float
        )

    # Replace zero sums with NaN to prevent division by zero
    C[np.isclose(C, 0)] = np.nan

    # Return the array closed to sum to 1
    return np.divide(X, C)


@overload
def renormalise(
    df: pd.DataFrame, components: list[str] | None, scale: float
) -> pd.DataFrame: ...
@overload
def renormalise(
    df: pd.Series, components: list[str] | None, scale: float
) -> pd.Series: ...
def renormalise(
    df: pd.DataFrame | pd.Series,
    components: list[str] | None = None,
    scale: float = 100.0,
) -> pd.DataFrame | pd.Series:
    """
    Renormalises compositional data to ensure closure.

    Parameters
    ------------
    df : pandas.DataFrame
        Dataframe to renomalise.
    components : list
        Option subcompositon to renormalise to 100. Useful for the use case
        where compostional data and non-compositional data are stored in the
        same dataframe.
    scale : float
        Closure parameter. Typically either 100 or 1.

    Returns
    --------
    pandas.DataFrame
        Renormalized dataframe.
    """
    if components is None:
        components = []
    dfc: pd.DataFrame | pd.Series = df.copy(deep=True)
    # Renormalise all columns if no components are specified
    if components:
        if not all(
            col in (dfc.columns if isinstance(dfc, pd.DataFrame) else dfc.index)
            for col in components
        ):
            raise ValueError("Not all specified components exist in the DataFrame.")
        dfc: pd.DataFrame | pd.Series = dfc[components]

    if (dfc <= 0).any().any():
        warnings.warn(
            "Non-positive entries found in specified components. "
            "Negative values have been replaced with NaN. "
            "Renormalisation assumes all positive entries.",
            UserWarning,
        )

    # Replace negative values with NaN
    dfc[dfc < 0] = np.nan
    # Handle division by zero by replacing zeros with NaN
    if isinstance(dfc, pd.DataFrame):
        return dfc.divide(dfc.sum(axis=1).replace(0, np.nan), axis=0) * scale
    else:
        sum_series: float = dfc.sum()
        if np.isclose(sum_series, 0):
            sum_series: float = np.nan
        return dfc.divide(sum_series)


def ALR(
    X: np.ndarray[tuple[int, int], np.dtype[np.number]], ind: int = -1, null_col=False
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Additive Log Ratio transformation.

    Parameters
    ---------------
    X: numpy.ndarray
        Array on which to perform the transformation, of shape `(N, D)`.
    ind: int
        Index of column used as denominator.
    null_col : bool
        Whether to keep the redundant column.

    Returns
    ---------
    numpy.ndarray
        ALR-transformed array, of shape `(N, D-1)`.
    """

    Y = X.copy()
    assert Y.ndim in [1, 2]
    dimensions = Y.shape[Y.ndim - 1]
    if ind < 0:
        ind += dimensions

    if Y.ndim == 2:
        Y = np.divide(Y, Y[:, ind][:, np.newaxis])
        if not null_col:
            Y = Y[:, [i for i in range(dimensions) if i != ind]]
    else:
        Y = np.divide(X, X[ind])
        if not null_col:
            Y = Y[[i for i in range(dimensions) if i != ind]]

    return np.log(Y)


def inverse_ALR(
    Y: np.ndarray, ind=-1, null_col=False
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Inverse Centred Log Ratio transformation.

    Parameters
    ---------------
    Y : numpy.ndarray
        Array on which to perform the inverse transformation, of shape `(N, D-1)`.
    ind : int
        Index of column used as denominator.
    null_col : bool, `False`
        Whether the array contains an extra redundant column
        (i.e. shape is `(N, D)`).

    Returns
    --------
    numpy.ndarray
        Inverse-ALR transformed array, of shape `(N, D)`.
    """
    assert Y.ndim in [1, 2]

    X = Y.copy()
    dimensions = X.shape[X.ndim - 1]
    if not null_col:
        idx = np.arange(0, dimensions + 1)

        if ind != -1:
            idx = np.array(list(idx[idx < ind]) + [-1] + list(idx[idx >= ind + 1] - 1))

        # Add a zero-column and reorder columns
        if Y.ndim == 2:
            X = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
            X = X[:, idx]
        else:
            X = np.append(X, np.array([0]))
            X = X[idx]

    # Inverse log and closure operations
    X = np.exp(X)
    X = close(X)
    return X


def CLR(X: np.ndarray) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Centred Log Ratio transformation.

    Parameters
    ---------------
    X : numpy.ndarray
        2D array on which to perform the transformation, of shape `(N, D)`.

    Returns
    ---------
    numpy.ndarray
        CLR-transformed array, of shape `(N, D)`.
    """
    X = np.array(X)
    X = np.divide(X, np.sum(X, axis=1).reshape(-1, 1))  # Closure operation
    Y = np.log(X)  # Log operation
    nvars = max(X.shape[1], 1)  # if the array is empty we'd get a div-by-0 error
    G = (1 / nvars) * np.nansum(Y, axis=1)[:, np.newaxis]
    Y -= G
    return Y


def inverse_CLR(Y: np.ndarray) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Inverse Centred Log Ratio transformation.

    Parameters
    ---------------
    Y : numpy.ndarray
        Array on which to perform the inverse transformation, of shape `(N, D)`.

    Returns
    ---------
    numpy.ndarray
        Inverse-CLR transformed array, of shape `(N, D)`.
    """
    # Inverse of log operation
    X = np.exp(Y)
    # Closure operation
    X = np.divide(X, np.nansum(X, axis=1)[:, np.newaxis])
    return X


def ILR(
    X: np.ndarray, psi: np.ndarray | None = None, **kwargs
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Isometric Log Ratio transformation.

    Parameters
    ---------------
    X : numpy.ndarray
        Array on which to perform the transformation, of shape `(N, D)`.
    psi : numpy.ndarray
        Array or matrix representing the ILR basis; optionally specified.

    Returns
    --------
    numpy.ndarray
        ILR-transformed array, of shape `(N, D-1)`.
    """
    d = X.shape[1]
    Y = CLR(X)
    if psi is None:
        psi = helmert_basis(D=d, **kwargs)  # Get a basis
    assert np.allclose(psi @ psi.T, np.eye(d - 1))
    return Y @ psi.T


def inverse_ILR(
    Y: np.ndarray, X: np.ndarray | None = None, psi: np.ndarray | None = None, **kwargs
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Inverse Isometric Log Ratio transformation.

    Parameters
    ---------------
    Y : numpy.ndarray
        Array on which to perform the inverse transformation, of shape `(N, D-1)`.
    X : numpy.ndarray, `None`
        Optional specification for an array from which to derive the orthonormal basis,
        with shape `(N, D)`.
    psi : numpy.ndarray
        Array or matrix representing the ILR basis; optionally specified.

    Returns
    --------
    numpy.ndarray
        Inverse-ILR transformed array, of shape `(N, D)`.
    """
    if psi is None:
        psi = helmert_basis(D=Y.shape[1] + 1, **kwargs)
    C = Y @ psi
    X = inverse_CLR(C)  # Inverse log operation
    return X


def logratiomean(df: pd.DataFrame, transform: Callable = CLR) -> pd.Series:
    """
    Take a mean of log-ratios along the index of a dataframe.

    Parameters
    -----------
    df : pandas.DataFrame
        Dataframe from which to compute a mean along the index.
    transform : Callable
        Log transform to use.
    inverse_transform : Callable
        Inverse of log transform.

    Returns
    ---------
    pandas.Series
        Mean values as a pandas series.
    """
    tfm, inv_tfm = get_transforms(transform)
    return pd.Series(
        inv_tfm(np.mean(tfm(df.values), axis=0)[np.newaxis, :])[0],
        index=df.columns,
    )


########################################################################################
# Logratio variable naming
########################################################################################


def _aggregate_sympy_constants(expr: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
    """
    Aggregate constants and symbolic components within a sympy expression to separate
    sub-expressions.

    Parameters
    -----------
    expr : sympy.core.expr.Expr
        Expression to aggregate. For matricies, use :func:`~sympy.Matrix.applyfunc`.

    Returns
    -------
    sympy.core.expr.Expr
    """
    const = expr.func(*[term for term in expr.args if not term.free_symbols])
    vars = expr.func(*[term for term in expr.args if term.free_symbols])
    if const:
        return sympy.UnevaluatedExpr(const) * sympy.UnevaluatedExpr(vars)
    else:
        return sympy.UnevaluatedExpr(vars)


def get_ALR_labels(
    df: pd.DataFrame, mode: str = "simple", ind: int = -1, **kwargs
) -> list[str]:
    """
    Get symbolic labels for ALR coordinates based on dataframe columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to generate ALR labels for.
    mode : str
        Mode of label to return (`LaTeX`, `simple`).

    Returns
    -------
    list
        List of ALR coordinates corresponding to dataframe columns.

    Notes
    ------
    Some variable names are protected in :mod:`sympy` and if used can result in errors.
    If one of these column names is found, it will be replaced with a title-cased
    duplicated version of itself (e.g. 'S' will be replaced by 'Ss').
    """

    names = [
        rf"{__sympy_protected_variables__.get(c, c)} / {df.columns[ind]}"
        for c in df.columns
    ]

    if mode.lower() == "latex":
        # edited to avoid issues with clashes between element names and latex (e.g. Ge)
        D = df.columns.size
        # encode symbolic variables
        vars = [sympy.var(f"c_{ix}") for ix in range(D)]
        expr = sympy.Matrix([[sympy.ln(v) for v in vars]])
        named_expr = expr.subs({k: v for (k, v) in zip(vars, names)})
        labels = [
            r"${}$".format(sympy.latex(l, mul_symbol="dot", ln_notation=True))
            for l in named_expr
        ]
    elif mode.lower() == "simple":
        labels = [f"ALR({n})" for n in names]
    else:
        msg = f"Label mode {mode} not recognised."
        raise NotImplementedError(msg)
    return labels


def get_CLR_labels(df: pd.DataFrame, mode: str = "simple", **kwargs) -> list[str]:
    """
    Get symbolic labels for CLR coordinates based on dataframe columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to generate CLR labels for.
    mode : str
        Mode of label to return (`LaTeX`, `simple`).

    Returns
    -------
    list
        List of CLR coordinates corresponding to dataframe columns.

    Notes
    ------
    Some variable names are protected in :mod:`sympy` and if used can result in errors.
    If one of these column names is found, it will be replaced with a title-cased
    duplicated version of itself (e.g. 'S' will be replaced by 'Ss').
    """

    names = [rf"{__sympy_protected_variables__.get(c, c)} / γ" for c in df.columns]
    D = df.columns.size

    if mode.lower() == "latex":
        # edited to avoid issues with clashes between element names and latex (e.g. Ge)
        D = df.columns.size
        # encode symbolic variables
        vars = [sympy.var(f"c_{ix}") for ix in range(D)]
        expr = sympy.Matrix([[sympy.ln(v) for v in vars]])
        named_expr = expr.subs({k: v for (k, v) in zip(vars, names)})
        labels = [
            r"${}$".format(sympy.latex(l, mul_symbol="dot", ln_notation=True))
            for l in named_expr
        ]
    elif mode.lower() == "simple":
        labels = [f"CLR({c}/G)" for c in df.columns]
    else:
        msg = f"Label mode {mode} not recognised."
        raise NotImplementedError(msg)
    return labels


def get_ILR_labels(df: pd.DataFrame, mode: str = "latex", **kwargs) -> list[str]:
    """
    Get symbolic labels for ILR coordinates based on dataframe columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to generate ILR labels for.
    mode : str
        Mode of label to return (`LaTeX`, `simple`).

    Returns
    -------
    list
        List of ILR coordinates corresponding to dataframe columns.

    Notes
    ------
    Some variable names are protected in :mod:`sympy` and if used can result in errors.
    If one of these column names is found, it will be replaced with a title-cased
    duplicated version of itself (e.g. 'S' will be replaced by 'Ss').
    """
    D = df.columns.size
    # encode symbolic variables
    sym_vars = [sympy.var(f"c_{ix}") for ix in range(D)]
    arr = sympy.Matrix([[sympy.ln(v) for v in sym_vars]])

    # this is the CLR --> ILR transform
    helmert = symbolic_helmert_basis(D, **kwargs)
    expr = sympy.simplify(
        sympy.logcombine(sympy.simplify(arr @ helmert.transpose()), force=True)
    )
    expr = expr.applyfunc(_aggregate_sympy_constants)
    # sub in Phi (the CLR normalisation variable)
    names = [rf"{__sympy_protected_variables__.get(c, c)} / γ" for c in df.columns]
    named_expr = expr.subs({k: v for (k, v) in zip(sym_vars, names)})
    # format latex labels
    if mode.lower() == "latex":
        labels = [
            r"${}$".format(sympy.latex(l, mul_symbol="dot", ln_notation=True))
            for l in named_expr
        ]
    elif mode.lower() == "simple":
        # here we could exclude scaling terms and just use ILR(A/B)
        unscaled_components = named_expr.applyfunc(
            lambda x: x.func(*[term for term in x.args if term.free_symbols])
        )
        labels = [str(l).replace("log", "ILR") for l in unscaled_components]
    else:
        msg = f"Label mode {mode} not recognised."
        raise NotImplementedError(msg)
    return labels


########################################################################################
# Box-cox transforms
########################################################################################


def boxcox(
    X: np.ndarray[tuple[int, int], np.dtype[np.number]] | pd.DataFrame,
    lmbda: float | None = None,
    lmbda_search_space: tuple[float, float] = (-1, 5),
    search_steps: int = 100,
    return_lmbda: bool = False,
) -> (
    np.ndarray[tuple[int, int], np.dtype[np.floating]]
    | pd.DataFrame
    | tuple[np.ndarray[tuple[int, int], np.dtype[np.floating]] | pd.DataFrame, float]
):
    """
    Box-Cox transformation.

    Parameters
    ---------------
    X : numpy.ndarray
        Array on which to perform the transformation.
    lmbda : flaot
        Lambda value used to forward-transform values. If none, it will be calculated
        using the mean.
    lmbda_search_space : tuple
        Range tuple (min, max).
    search_steps : int
        Steps for lambda search range.
    return_lmbda : bool
        Whether to also return the lambda value.

    Returns
    -------
    numpy.ndarray | pandas.DataFrame | tuple[np.ndarray | pandas.DataFrame,float]
        Box-Cox transformed array. If `return_lmbda` is true, tuple contains data and
        lambda value.
    """
    if isinstance(X, pd.DataFrame):
        _X = X.values
    else:
        _X = X.copy()

    if lmbda is None:
        l_search = np.linspace(*lmbda_search_space, search_steps)
        llf = np.vstack(
            [scipy.stats.boxcox_llf(ix, _X.T) for ix in l_search]
        ).T  # nsamples, n_search_steps
        if llf.shape[0] == 1:
            mean_llf = llf[0]
        else:
            mean_llf = np.nansum(llf, axis=0)

        lmbda = l_search[mean_llf == np.nanmax(mean_llf)]

    if _X.ndim < 2:
        out = scipy.stats.boxcox(_X, lmbda)
    elif _X.shape[0] == 1:
        out = scipy.stats.boxcox(np.squeeze(_X), lmbda)
    else:
        out = np.apply_along_axis(scipy.stats.boxcox, 0, _X, lmbda)

    if isinstance(_X, pd.DataFrame):
        _out = X.copy()
        _out.loc[:, :] = out
        out = _out

    if return_lmbda:
        return out, lmbda
    else:
        return out


def inverse_boxcox(
    Y: np.ndarray[tuple[int, int], np.dtype[np.floating]], lmbda: float
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Inverse Box-Cox transformation.

    Parameters
    ---------------
    Y : numpy.ndarray
        Array on which to perform the transformation.
    lmbda : float
        Lambda value used to forward-transform values.

    Returns
    -------
    numpy.ndarray
        Inverse Box-Cox transformed array.
    """
    return scipy.special.inv_boxcox(Y, lmbda)


########################################################################################
# Functions for spherical coordinate transformation of compositional data.
########################################################################################
"""
The functions below were derived from the references below, but should be in line
with the work which preceeded them.

Neocleous, T., Aitken, C., Zadora, G., 2011. Transformations for compositional data
with zeros with an application to forensic evidence evaluation. Chemometrics and
Intelligent Laboratory Systems 109, 77-85. https://doi.org/10.1016/j.chemolab.2011.08.003

Wang, H., Liu, Q., Mok, H.M.K., Fu, L., Tse, W.M., 2007. A hyperspherical transformation
forecasting model for compositional data. European Journal of Operational Research 179,
459-468. https://doi.org/10.1016/j.ejor.2006.03.039
"""


def sphere(ys: np.ndarray[tuple[int, int], np.dtype[np.number]]):
    r"""
    Spherical coordinate transformation for compositional data.

    Parameters
    ----------
    ys : numpy.ndarray
        Compositional data to transform (shape (n, D)).

    Returns
    -------
    θ : numpy.ndarray
        Array of angles in radians (:math:`(0, \pi / 2]`)

    Notes
    -----
    :func:`numpy.arccos` will return angles in the range :math:`(0, \pi)`. This shouldn't be
    an issue for this function given that the input values are all positive.
    """
    p = ys.shape[1] - 1
    _ys = np.sqrt(close(ys))  # closure operation
    θ = np.ones((ys.shape[0], p))

    indicies = np.arange(1, p + 1)[::-1]
    for ix in indicies:  # we have to recurse from p back down to #2
        if ix == p:
            S = 1
        else:
            # vector - the product of sin components
            S = np.prod(np.sin(θ[:, ix:]), axis=1)
            # where this evaluates to zero, the composition is all in the first component
            S[np.isclose(S, 0.0)] = 1.0

        ratios = _ys[:, ix] / S
        # where this looks like it could be slightly higher than 1
        # np.arcos will return np.nan, so we can filter these.
        ratios[np.isclose(ratios, 1.0)] = 1.0
        θ[:, ix - 1] = np.arccos(ratios)
    return θ


def inverse_sphere(θ: np.ndarray[tuple[int, int], np.dtype[np.floating]]):
    """
    Inverse spherical coordinate transformation to revert back to compositional data
    in the simplex.

    Parameters
    ----------
    θ : numpy.ndarray
        Angular coordinates to revert.

    Returns
    -------
    ys : numpy.ndarray
        Compositional (simplex) coordinates, normalised to 1.
    """
    p = θ.shape[1]
    θ.shape[0]
    y = np.ones((θ.shape[0], p + 1)) * np.pi / 2

    sinθ: np.ndarray[tuple[int, int], np.dtype[np.floating]] = np.sin(θ)
    cosθ: np.ndarray[tuple[int, int], np.dtype[np.floating]] = np.cos(θ)

    indicies = np.arange(0, p + 1)
    C: float
    for ix in indicies:
        if ix == 0:
            C = 1.0
        else:
            C = cosθ[:, ix - 1]

        if ix == p:
            S = 1.0
        else:
            S = np.prod(sinθ[:, ix:], axis=1)
        y[:, ix] = C * S

    return y**2


################################################################################


def compositional_cosine_distances(
    arr: np.ndarray[tuple[int, int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Calculate a distance matrix corresponding to the angles between a number
    of compositional vectors.

    Parameters
    ----------
    arr: numpy.ndarray
        Array of n-dimensional compositions of shape (n_samples, n).

    Returns
    -------
    numpy.ndarray
        Array of angular distances of shape (n_samples, n_samples).
    """
    # all vectors are unit vectors where we start with closed compositions
    _closed = close(arr)
    # and we can then calculate the cosine similarity
    cosine_sim = np.dot(
        np.sqrt(np.expand_dims(_closed, axis=1)),
        np.sqrt(np.expand_dims(_closed, axis=2)),
    ).squeeze()
    # finally, we convert the cosines back to angules
    return np.arccos(np.clip(cosine_sim, -1.0, 1.0))


########################################################################################
# Meta-functions for accessing transformations.
########################################################################################


def get_transforms(name: str | Callable) -> tuple[Callable, Callable]:
    """
    Lookup a transform-inverse transform pair by name.

    Parameters
    ----------
    name : str
        Name of of the transform pairs (e.g. 'CLR').

    Returns
    -------
    tfm, inv_tfm : tuple[Callable,Callable]
        Transform and inverse transform functions.
    """
    if callable(name):  #  callable
        name = name.__name__

    tfm, inv_tfm = __TRANSFORMS__.get(name)
    return tfm, inv_tfm


def _load_transforms():
    """
    Load the transform pairs into the module level variable for later lookup.

    Returns
    -------
    dict
    """
    return {
        f: (globals().get(f), globals().get(f"inverse_{f}"))
        for f in globals()
        if f"inverse_{f}" in globals()
    }


__TRANSFORMS__.update(_load_transforms())
