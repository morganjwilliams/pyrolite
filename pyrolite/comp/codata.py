import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from ..util.math import orthagonal_basis
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def close(X: np.ndarray):
    if X.ndim == 2:
        return np.divide(X, np.sum(X, axis=1)[:, np.newaxis])
    else:
        return np.divide(X, np.sum(X, axis=0))


def renormalise(df: pd.DataFrame, components:list=[], scale=100.):
    """
    Renormalises compositional data to ensure closure.

    Parameters:
    ------------
    df: pd.DataFrame
        Dataframe to renomalise.
    components: list
        Option subcompositon to renormalise to 100. Useful for the use case
        where compostional data and non-compositional data are stored in the
        same dataframe.
    scale: float, 100.
        Closure parameter. Typically either 100 or 1.
    """
    dfc = df.copy()
    if components:
        cmpnts = [c for c in components if c in dfc.columns]
        dfc.loc[:, cmpnts] =  scale * dfc.loc[:, cmpnts].divide(
                              dfc.loc[:, cmpnts].sum(axis=1).replace(0, np.nan),
                                                               axis=0)
        return dfc
    else:
        dfc = dfc.divide(dfc.sum(axis=1).replace(0, 100.), axis=0) * scale
        return dfc


def additive_log_ratio(X: np.ndarray, ind: int=-1):
    """
    Inverse Additive Log Ratio transformation.

    Parameters:
    ---------------
    X: np.ndarray
        Array on which to perform the inverse transformation.
    ind: int
        Index of column used as denominator.
    """

    Y = X.copy()
    assert Y.ndim in [1, 2]
    dimensions = Y.shape[Y.ndim-1]
    if ind < 0: ind += dimensions

    if Y.ndim == 2:
        Y = np.divide(Y, Y[:, ind][:, np.newaxis])
        Y = np.log(Y[:, [i for i in range(dimensions) if not i==ind]])
    else:
        Y = np.divide(X, X[ind])
        Y = np.log(Y[[i for i in range(dimensions) if not i==ind]])

    return Y


def inverse_additive_log_ratio(Y: np.ndarray, ind=-1):
    """
    Inverse Centred Log Ratio transformation.

    Parameters:
    ---------------
    X: np.ndarray
        Array on which to perform the inverse transformation.
    ind: int
        Index of column used as denominator.
    """
    assert Y.ndim in [1, 2]

    X = Y.copy()
    dimensions = X.shape[X.ndim-1]
    idx = np.arange(0, dimensions+1)

    if ind != -1:
        idx = np.array(list(idx[idx < ind]) +
                       [-1] +
                       list(idx[idx >= ind+1]-1))

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


def alr(*args, **kwargs):
    """
    Short form of Additive Log Ratio transformation.

    Parameters:
    ---------------
    Y: np.ndarray
        Array on which to perform the inverse transformation.
    ind: int
        Index of column used as denominator.
    """
    return additive_log_ratio(*args, **kwargs)


def inv_alr(*args, **kwargs):
    """
    Short form of Inverse Additive Log Ratio transformation.

    Parameters:
    ---------------
    Y: np.ndarray
        Array on which to perform the inverse transformation.
    ind: int
        Index of column used as denominator.
    """
    return inverse_additive_log_ratio(*args, **kwargs)


def clr(X: np.ndarray):
    """
    Centred Log Ratio transformation.

    Parameters:
    ---------------
    X: np.ndarray
        Array on which to perform the transformation.
    """
    X = np.divide(X, np.sum(X, axis=1)[:, np.newaxis])  # Closure operation
    Y = np.log(X)  # Log operation
    Y -= 1/X.shape[1] * np.nansum(Y, axis=1)[:, np.newaxis]
    return Y


def inv_clr(Y: np.ndarray):
    """
    Inverse Centred Log Ratio transformation.

    Parameters:
    ---------------
    Y: np.ndarray
        Array on which to perform the inverse transformation.
    """
    # Inverse of log operation
    X = np.exp(Y)
    # Closure operation
    X = np.divide(X, np.nansum(X, axis=1)[:, np.newaxis])
    return X


def ilr(X: np.ndarray):
    """
    Isotmetric Log Ratio transformation.

    Parameters:
    ---------------
    X: np.ndarray
        Array on which to perform the transformation.
    """
    d = X.shape[1]
    Y = clr(X)
    psi = orthagonal_basis(X)  # Get a basis
    psi = orthagonal_basis(clr(X)) # trying to get right algorithm
    assert np.allclose(psi @ psi.T, np.eye(d-1))
    return Y @ psi.T


def inv_ilr(Y: np.ndarray, X: np.ndarray=None):
    """
    Inverse Isometric Log Ratio transformation.

    Parameters:
    ---------------
    Y: np.ndarray
        Array on which to perform the inverse transformation.
    """
    psi = orthagonal_basis(X)
    C = Y @ psi
    X = inv_clr(C)  # Inverse log operation
    return X


class LinearTransform(TransformerMixin):
    """
    Linear Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = 'Crude'

    def transform(self, X, *args):
        X = np.array(X)
        return X

    def inverse_transform(self, Y, *args):
        Y = np.array(Y)
        return Y

    def fit(self, X, *args):
        return self


class ALRTransform(TransformerMixin):
    """
    Additive Log Ratio Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = 'ALR'

    def transform(self, X, *args, **kwargs):
        X = np.array(X)
        return alr(X, *args, **kwargs)

    def inverse_transform(self, Y, *args, **kwargs):
        Y = np.array(Y)
        return inv_alr(Y, *args, **kwargs)

    def fit(self, X, *args, **kwargs):
        return self


class CLRTransform(TransformerMixin):
    """
    Centred Log Ratio Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = 'CLR'

    def transform(self, X, *args, **kwargs):
        X = np.array(X)
        return clr(X, *args, **kwargs)

    def inverse_transform(self, Y, *args, **kwargs):
        Y = np.array(Y)
        return inv_clr(Y, *args, **kwargs)

    def fit(self, X, *args, **kwargs):
        return self


class ILRTransform(TransformerMixin):
    """
    Isometric Log Ratio Transformer for scikit-learn like use.
    """

    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = 'ILR'

    def transform(self, X, *args, **kwargs):
        X = np.array(X)
        self.X = X
        return ilr(X, *args, **kwargs)

    def inverse_transform(self, Y, *args, **kwargs):
        Y = np.array(Y)
        return inv_ilr(Y, X=self.X, *args, **kwargs)

    def fit(self, X, *args, **kwargs):
        return self
