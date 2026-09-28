from numpy.f2py.crackfortran import n
import warnings

import numpy as np
import pandas as pd

from ..util.log import Handle
from .codata import ALR, inverse_ALR

logger = Handle(__name__)


def get_full_column(X: np.ndarray) -> int:
    """
    Returns the index of the first array column which contains only finite
    numbers (i.e. no missing data, nan, inf).

    Parameters
    ---------------
    X : numpy.ndarray
        Array for which to find the first full column within.

    Returns
    --------
    int
        Index of the first full column.
    """
    if len(X.shape) == 1:
        X = X.reshape((1, *X.shape))
    inds = np.arange(X.shape[1])
    wherenonnan = np.isfinite(X).all(axis=0)
    ind = inds[wherenonnan][0]
    return ind


def weights_from_array(X: np.ndarray) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    """
    Returns a set of equal weights with size equal to that of the first axis of
    an array.

    Parameters
    ---------------
    X : numpy.ndarray
        Array of compositions to produce weights for.

    Returns
    --------
    numpy.ndarray
        Array of weights.
    """
    wts = np.ones(X.shape[0], dtype=np.float32)
    return wts / np.sum(wts)


def nan_weighted_mean(
    X: np.ndarray[tuple[int, int], np.dtype[np.number]],
    weights: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None,
) -> np.ndarray[tuple[int, int]]:
    """
    Returns a weighted mean of compositions, where weights are renormalised
    to account for missing data.

    Parameters
    ---------------
    X : numpy.ndarray
        Array of compositions to take a weighted mean of.
    weights : numpy.ndarray
        Array of weights.

    Returns
    --------
    numpy.ndarray
        Array mean.
    """
    if weights is None:
        weights = weights_from_array(X)
    weights = np.array(weights) / np.nansum(weights)

    mask = (np.isnan(X) + np.isinf(X)) > 0
    if not mask.any():
        return np.average(X, weights=weights, axis=0)
    else:
        return np.ma.average(np.ma.array(X, mask=mask), weights=weights, axis=0)


def compositional_mean(
    df: pd.DataFrame,
    weights: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None,
    **kwargs,
) -> pd.Series:
    """
    Implements an aggregation using a compositional weighted mean.

    Parameters
    ---------------
    df : pandas.DataFrame
        Dataframe of compositions to aggregate.
    weights : numpy.ndarray
        Array of weights.

    Returns
    --------
    pandas.Series
        Mean values along index of dataframe.
    """
    if weights is None:
        weights = []
    non_nan_cols = df.dropna(axis=1, how="all").columns
    assert not df.loc[:, non_nan_cols].isna().values.any()
    mean = df.iloc[0, :].copy()
    if not weights:
        weights = np.ones(len(df.index.values))
    weights = np.array(weights) / np.nansum(weights)

    logmean = ALR(df.loc[:, non_nan_cols].values).T @ weights[:, np.newaxis]
    # this renormalises by default
    mean.loc[non_nan_cols] = inverse_ALR(logmean.T.squeeze())
    return mean


def nan_weighted_compositional_mean(
    X: np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.number]],
    weights: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None,
    ind: int | None = None,
    renorm: bool = True,
    **kwargs,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """
    Implements an aggregation using a weighted mean, but accounts
    for nans. Requires at least one non-nan column for ALR mean.

    When used for internal standardisation, there should be only a single
    common element - this would be used by default as the divisor here. When
    used for multiple-standardisation, the [specified] or first common
    element will be used.

    Parameters
    ---------------
    X : numpy.ndarray
        Array of compositions to aggregate.
    weights : numpy.ndarray
        Array of weights.
    ind : int
        Index of the column to use as the ALR divisor.
    renorm : bool
        Whether to renormalise the output compositional mean to unity.

    Returns
    -------
    numpy.ndarray
        An array with the mean composition.
    """
    if X.ndim == 1:  # if it's a single row
        return X
    else:
        if weights is None:
            weights = weights_from_array(X)
        else:
            weights = np.array(weights) / np.sum(weights, axis=-1)

        if ind is None:  # take the first column which has no nans
            ind = get_full_column(X)

        if X.ndim < 3 and X.shape[0] == 1:
            div = X[:, ind].squeeze()  # check this
        else:
            div = X[:, ind].squeeze()[:, np.newaxis]

        logvals = np.log(np.divide(X, div))
        mean = np.nan * np.ones(X.shape[1:])

        ixs = np.arange(logvals.shape[1])
        if X.ndim == 2:
            indexes = ixs
        elif X.ndim == 3:
            iys = np.arange(logvals.shape[2])
            indexes = np.ixs_(ixs, iys)

        mean[indexes] = nan_weighted_mean(logvals[:, indexes], weights=weights)

        mean = np.exp(mean.squeeze())
        if renorm:
            mean /= np.nansum(mean)
        return mean


def cross_ratios(
    df: pd.DataFrame,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.floating]]:
    """
    Takes ratios of values across a dataframe, such that columns are
    denominators and the row indexes the numerators, to create a square array.
    Returns one array per record.

    Parameters
    ---------------
    df : pandas.DataFrame
        Dataframe of compositions to create ratios of.

    Returns
    -------
    numpy.ndarray
        A 3D array of ratios.
    """
    ratios = np.ones((len(df.index), len(df.columns), len(df.columns)))
    for idx in range(df.index.size):
        row_vals = df.iloc[idx, :].values
        r1 = row_vals.T[:, np.newaxis] @ np.ones_like(row_vals)[np.newaxis, :]
        ratios[idx] = r1 / r1.T
    return ratios


def np_cross_ratios(
    X: np.ndarray, debug: bool = False
) -> np.ndarray[tuple[int, int, int], np.dtype[np.floating]]:
    """
    Takes ratios of values across an array, such that columns are
    denominators and the row indexes the numerators, to create a square array.
    Returns one array per record.

    Parameters
    ---------------
    X : numpy.ndarray
        Array of compositions to create ratios of.

    Returns
    -------
    numpy.ndarray
        A 3D array of ratios.
    """
    X = X.copy()
    with warnings.catch_warnings():
        # can get invalid values which raise RuntimeWarnings
        # consider changing to np.errstate
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X[X <= 0] = np.nan
    if X.ndim == 1:
        index_length = 1
        X = X.reshape((1, *X.shape))
    else:
        index_length = X.shape[0]
    dims = X.shape[-1]
    ratios = np.ones((index_length, dims, dims))
    for idx in range(index_length):
        row_vals = X[idx, :]
        r1 = row_vals.T[:, np.newaxis] @ np.ones_like(row_vals)[np.newaxis, :]
        ratios[idx] = r1.T / r1

    if debug:
        try:
            diags = ratios[:, np.arange(dims), np.arange(dims)]
            # check all diags are 1.
            assert np.allclose(diags, 1.0)
        except AssertionError:
            # check all diags are 1. or nan
            assert np.allclose(diags[~np.isnan(diags)], 1.0)

    return ratios


def standardise_aggregate(
    df: pd.DataFrame,
    int_std: str | None = None,
    fixed_record_idx: int = 0,
    renorm: bool = True,
    **kwargs,
) -> pd.Series:
    """
    Performs internal standardisation and aggregates dissimilar geochemical
    records. Note: this changes the closure parameter, and is generally
    intended to integrate major and trace element records.

    Parameters
    ---------------
    df : pandas.DataFrame
        Dataframe of compositions to aggregate of.
    int_std : str
        Name of the internal standard column.
    fixed_record_idx : int
        Numeric index of a specific record's for which to retain the internal
        standard value (e.g for standardising trace element data).
    renorm : bool
        Whether to renormalise to unity.

    Returns
    -------
    pandas.Series
        A series representing the internally standardised record.
    """
    if df.index.size == 1:  # catch single records
        return df.iloc[0]
    else:
        if int_std is None:
            # Get the 'internal standard column'
            potential_int_stds = df.count()[df.count() == df.count().max()].index.values
            assert len(potential_int_stds) > 0
            # Use an internal standard
            int_std = potential_int_stds[0]
            if len(potential_int_stds) > 1:
                logger.info("Multiple int. stds possible. Using " + str(int_std))

        non_nan_cols = df.dropna(axis=1, how="all").columns
        assert len(non_nan_cols)
        mean = nan_weighted_compositional_mean(
            df.values, ind=df.columns.get_loc(int_std), renorm=False
        )
        ser = pd.Series(mean, index=df.columns)
        multiplier = (
            df.iloc[fixed_record_idx, df.columns.get_loc(int_std)] / ser[int_std]
        )
        ser *= multiplier
        if renorm:
            ser /= np.nansum(ser.values)
        return ser
