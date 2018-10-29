import numpy as np
import pandas as pd


def impute_ratios(ratios: pd.DataFrame):
    """
    Pandas version of ratio matrix imputation.
    """
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


def np_impute_ratios(ratios: np.ndarray):
    """
    Numpy version of ratio matrix imputation.
    """
    finite = np.isfinite(ratios)
    not_finite = ~finite
    if not_finite.any():
        where_not_finite = np.argwhere(not_finite)
        print(where_not_finite)
        _ixs, _iys = where_not_finite.T
        ixs = _ixs[~(_ixs == _iys)]
        iys = _iys[~(_ixs == _iys)]
        where_not_finite = np.stack((ixs, iys)).T
        excludes = np.empty((ixs.size, ratios.shape[0]-2)).astype(int)
        indicies = np.arange(ratios.shape[0]).astype(int)
        for enm_ix in np.arange(ixs.size):
            excludes[enm_ix] = np.setdiff1d(indicies, where_not_finite[enm_ix])

        for enm_ix in np.arange(ixs.size):
            ex = excludes[enm_ix]
            ix, iy = where_not_finite[enm_ix].T
            ratios[ix, iy] = np.nanmean(ratios[ix, ex] + ratios[ex, iy])
    return ratios
