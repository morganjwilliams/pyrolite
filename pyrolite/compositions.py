from types import MethodType
import numpy as np
import pandas as pd
import scipy
from sklearn.base import TransformerMixin
import logging
import warnings
log = logging.getLogger(__name__)

def close(X: np.ndarray):
    if X.ndim == 2:
        return np.divide(X, np.sum(X, axis=1)[:, np.newaxis])
    else:
        return np.divide(X, np.sum(X, axis=0))


def compositional_mean(df, weights=[], **kwargs):
    """
    Implements an aggregation using a weighted mean.
    """
    non_nan_cols = df.dropna(axis=1, how='all').columns
    assert not df.loc[:, non_nan_cols].isna().values.any()
    mean = df.iloc[0, :].copy()
    if not weights:
        weights = np.ones(len(df.index.values))
    weights = np.array(weights)/np.nansum(weights)

    logmean = alr(df.loc[:, non_nan_cols].values).T @ weights[:, np.newaxis]
    mean.loc[non_nan_cols] = inv_alr(logmean.T.squeeze()) # this renormalises by default
    return mean


def nan_weighted_mean(arr:np.ndarray, weights=None,):
    if weights is None:
        weights = weights or weights_from_array(arr)

    #if arr.ndim == 1: arr = arr.reshape((1, *arr.shape))
    #if weights.ndim == 1: weights = weights.reshape((*weights.shape, 1))

    if np.isnan(arr).any():
        mean = np.nanmean(arr, axis=0)
        if not (weights == weights[0]).all():  # if weights needed
            cs = np.arange(arr.shape[1])
            nonnan_idx = np.nonzero(~np.isnan(arr[:, cs]))
            if len(nonnan_idx[0]):  # if there are any non-nan elements
                c_weights = weights.copy()
                c_weights = c_weights[nonnan_idx] / \
                            c_weights[nonnan_idx].sum()
                mean[c] = arr[:, cs][nonnan_idx] @ c_weights
    else:
        mean = arr.T @ weights
        mean = mean.T.squeeze()
    mean = mean.reshape(arr.shape[1:]) # this should be compatible
    return mean


def get_nonnan_column(arr:np.ndarray):
    """Returns the first column without nans in it."""
    if len(arr.shape)==1:
        arr = arr.reshape((1, *arr.shape))
    inds = np.arange(arr.shape[1])
    wherenonnan = ~np.isnan(arr).any(axis=0)
    ind = inds[wherenonnan][0]
    return ind


def weights_from_array(arr:np.ndarray):
    """
    Returns a set of equal weights for components
    along the first axis of an array.
    """
    wts = np.ones((arr.shape[0]))
    wts = wts/np.sum(wts)
    wts = wts.T
    return wts


def nan_weighted_compositional_mean(arr: np.ndarray,
                                    weights=None,
                                    ind=None,
                                    renorm=True,
                                    **kwargs):
    """
    Implements an aggregation using a weighted mean, but accounts
    for nans. Requires at least one non-nan column for alr mean.

    When used for internal standardisation, there should be only a single
    common element - this would be used by default as the divisor here.

    When used for multiple-standardisation, the [specified] or first common
    element will be used.

    Input array has analyses along the first axis.
    """
    if arr.ndim == 1: #if it's a single row
        return arr
    else:
        if weights is None:
            weights = weights_from_array(arr)
        else:
            weights = np.array(weights)/np.sum(weights, axis=-1)

        if weights.ndim == 1:
            weights = weights.reshape((*weights.shape, 1))

        if ind is None:  # take the first column which has no nans
            ind = get_nonnan_column(arr)

        if arr.ndim < 3 and arr.shape[0] == 1:
            div = arr[:, ind].squeeze() # check this
        else:
            div = arr[:, ind].squeeze()[:, np.newaxis]

        logvals = np.log(np.divide(arr, div))
        mean = np.nan * np.ones(arr.shape[1:])

        ixs = np.arange(logvals.shape[1])
        if arr.ndim == 2:
            indexes = ixs
        elif arr.ndim == 3:
            iys = np.arange(logvals.shape[2])
            indexes = np.ixs_(ixs, iys)

        mean[indexes] = nan_weighted_mean(logvals[:, indexes],
                                          weights=weights)

        mean = np.exp(mean.squeeze())
        if renorm: mean /= np.nansum(mean)
        return mean


def cross_ratios(df: pd.DataFrame):
    """
    Takes ratios of values across a a dataframe to create a square array,
    such that columns are denominators and the row indexes the numerators.
    Returns an array arrays (one per dataframe record).
    """
    ratios = np.ones((len(df.index), len(df.columns), len(df.columns)))
    for idx in df.index:
        row_vals = df.loc[idx, :].values
        r1 = row_vals.T[:, np.newaxis] @ np.ones_like(row_vals)[np.newaxis, :]
        ratios[idx] = r1 / r1.T
    return ratios


def impute_ratios(ratios: pd.DataFrame):
    for IS in ratios.columns:
        ser = ratios.loc[:,  IS]
        if ser.isnull().any():
            non_null_idxs = ser.loc[~ser.isnull()].index.values
            null_idxs = ser.loc[ser.isnull()].index.values
            for null in null_idxs:
                # e.g.  Ti / MgO = Ti/SiO2 * SiO2 / MgO
                inverse_ratios = ratios.loc[null, non_null_idxs] # e.g. SiO2/MgO ratios
                non_null_ISratios = ratios.loc[non_null_idxs, IS] # e.g. Ti/SiO2 ratios
                predicted_ratios = inverse_ratios * non_null_ISratios
                ratios.loc[null, IS] = np.exp(np.nanmean(np.log(predicted_ratios)))
    return ratios


def standardise_aggregate(df: pd.DataFrame,
                          int_std=None,
                          fixed_record_idx=0,
                          renorm=True,
                          **kwargs):
    """
    Performs internal standardisation and aggregates dissimilar geochemical records.
    Note: this changes the closure parameter, and is generally intended to integrate
    major and trace element records.
    """
    if df.index.size == 1: # catch single records
        return df
    else:
        if int_std is None:
            # Get the 'internal standard column'
            potential_int_stds = df.count()[df.count()==df.count().max()].index.values
            assert len(potential_int_stds) > 0
            # Use an internal standard
            int_std = potential_int_stds[0]
            if len(potential_int_stds) > 1:
                warnings.warn(f'Multiple int. stds possible. Using {int_std}')

        non_nan_cols = df.dropna(axis=1, how='all').columns
        assert len(non_nan_cols)
        mean = nan_weighted_compositional_mean(df.values,
                                        ind=list(df.columns).index(int_std),
                                        renorm=False)
        ser = pd.Series(mean, index=df.columns)

        ser = ser * df.loc[df.index[fixed_record_idx], int_std]
        if renorm: ser /= np.nansum(ser.values)
        return ser


def complex_standardise_aggregate(df, fix_int_std=None, renorm=True, fixed_record_idx=0):

    if fix_int_std is None:
        # create a n x d x d matrix for aggregating ratios
        ratios = cross_ratios(df)
        mean_ratios = pd.DataFrame(np.exp(np.nanmean(np.log(ratios), axis=0)),
                                   columns=df.columns,
                                   index=df.columns)
        # Filling in the null values in a ratio matrix
        mean_ratios = impute_ratios(mean_ratios)
        # Here all internal standards are equal, we simply pick the first.
        IS = df.columns[0]
        mean = np.exp(np.mean(np.log(mean_ratios/mean_ratios.loc[IS, :]), axis=1))
        mean /= np.nansum(mean.values) # This needs to be renormalised to make logical sense
        return mean
    else:
        # fallback to internal standardisation
        return standardise_aggregate(df,
                                     int_std=fix_int_std,
                                     fixed_record_idx=fixed_record_idx,
                                     renorm=renorm)


def nancov(X, method='replace'):
    """
    Generates a covariance matrix excluding nan-components.
    Done on a column-column/pairwise basis.
    The result Y may not be a positive definite matrix.
    """
    if method=='rowexclude':
        Xnanfree = X[np.all(np.isfinite(X), axis=1), :].T
        #assert Xnanfree.shape[1] > Xnanfree.shape[0]
        #(1/m)X^T*X
        return np.cov(Xnanfree)
    else:
        X = np.array(X, ndmin=2, dtype=float)
        X -= np.nanmean(X, axis=0)#[:, np.newaxis]
        cov = np.empty((X.shape[1], X.shape[1]))
        cols = range(X.shape[1])
        for n in cols:
            for m in [i for i in cols if i>=n] :
                fn = np.isfinite(X[:, n])
                fm = np.isfinite(X[:, m])
                if method=='replace':
                    X[~fn, n] = 0
                    X[~fm, m] = 0
                    fact = fn.shape[0] - 1
                    c= np.dot(X[:, n], X[:, m])/fact
                else:
                    f = fn & fm
                    fact = f.shape[0] - 1
                    c = np.dot(X[f, n], X[f, m])/fact
                cov[n, m] = c
                cov[m, n] = c
        return cov


def renormalise(df: pd.DataFrame, components:list=[], scale=100.):
    """
    Renormalises compositional data to ensure closure.
    A subset of components can be used for flexibility.
    For data which sums to 0, 100 is returned - e.g. for TE-only datasets
    """
    dfc = df.copy()
    if components:
        cmpnts = [c for c in components if c in dfc.columns]
        dfc.loc[:, cmpnts] =  scale * dfc.loc[:, cmpnts].divide(
                              dfc.loc[:, cmpnts].sum(axis=1).replace(0, np.nan),
                                                               axis=0)
        return dfc
    else:
        dfc = dfc.divide(dfc.sum(axis=1).replace(0, 100), axis=0) * scale
        return dfc


def additive_log_ratio(X: np.ndarray, ind: int=-1):
    """Additive log ratio transform. """

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
    Inverse additive log ratio transform.
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
    return additive_log_ratio(*args, **kwargs)


def inv_alr(*args, **kwargs):
    return inverse_additive_log_ratio(*args, **kwargs)


def clr(X: np.ndarray):
    X = np.divide(X, np.sum(X, axis=1)[:, np.newaxis])  # Closure operation
    Y = np.log(X)  # Log operation
    Y -= 1/X.shape[1] * np.nansum(Y, axis=1)[:, np.newaxis]
    return Y


def inv_clr(Y: np.ndarray):
    X = np.exp(Y)  # Inverse of log operation
    X = np.divide(X, np.nansum(X, axis=1)[:, np.newaxis])  #Closure operation
    return X


def orthagonal_basis(X: np.ndarray):
    D = X.shape[1]
    H = scipy.linalg.helmert(D, full=False)  # D-1, D Helmert matrix, exact representation of ψ as in Egozogue's book
    return H[::-1]


def ilr(X: np.ndarray):
    d = X.shape[1]
    Y = clr(X)
    psi = orthagonal_basis(X)  # Get a basis
    psi = orthagonal_basis(clr(X)) # trying to get right algorithm
    assert np.allclose(psi @ psi.T, np.eye(d-1))
    return Y @ psi.T


def inv_ilr(Y: np.ndarray, X: np.ndarray=None):
    psi = orthagonal_basis(X)
    C = Y @ psi
    X = inv_clr(C)  # Inverse log operation
    return X


class LinearTransform(TransformerMixin):
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
