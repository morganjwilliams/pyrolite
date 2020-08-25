import numpy as np
import pandas as pd
import scipy.special
from collections import defaultdict


def md_pattern(Y):
    """
    Get the missing data patterns from an array.

    Parameters
    ------------
    Y : :class:`numpy.ndarray` | :class:`pandas.DataFrame`
        Input dataset.

    Returns
    ---------
    pattern_ids : :class:`numpy.ndarray`
        Pattern ID array.
    pattern_dict : :class:`dict`
        Dictionary of patterns indexed by pattern IDs. Contains a pattern and count
        for each pattern ID.
    """
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    N, D = Y.shape
    # use int64 for higher-D arrays
    pID = np.zeros(N).astype('int64')
    Ymiss = ~np.isfinite(Y)
    rows = np.arange(N)[~np.isfinite(np.sum(Y, axis=1))]
    max_pats = scipy.special.comb(D, np.arange(0, D + 1)).sum().astype('int64')
    pID[rows] = max_pats * 5  # initialise to high value
    pD = defaultdict(dict)

    pindex = 0  # 0 = no missing data
    pD[int(0)] = {"pattern": np.zeros(D).astype(bool), "freq": np.sum(pID == 0)}
    indexes = np.arange(N).astype(int)
    indexes = indexes[pID[indexes] > pindex]  # only look at md rows
    for idx in indexes:
        if pID[idx] > pindex:  # has missing data
            pindex += 1
            pID[idx] = pindex
            pattern = Ymiss[idx, :]
            pD[int(pindex)] = {"pattern": pattern, "freq": 0}
            if idx < N:
                _rix = np.arange(idx + 1, N)
                to_compare = _rix[pID[_rix] > pindex]
                where_same = to_compare[(Ymiss[to_compare, :] == pattern).all(axis=1)]
                pID[where_same] = pindex
    for ID in np.unique(pID).astype(int):
        pD[ID]["freq"] = np.sum(pID == ID)
    return pID, pD


def cooccurence_pattern(Y, normalize=False, log=False):
    """
    Get the co-occurence patterns from an array.

    Parameters
    ------------
    Y : :class:`numpy.ndarray` | :class:`pandas.DataFrame`
        Input dataset.
    normalize : :class:`bool`
        Whether to normalize the cooccurence to compare disparate variables.
    log : :class:`bool`
        Whether to take the log of the cooccurence.

    Returns
    ---------
    co_occur : :class:`numpy.ndarray`
        Cooccurence frequency array.
    """
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    _Y = Y.copy()
    _Y[~np.isfinite(_Y)] = 0
    _Y[_Y > 0] = 1
    _Y = _Y.astype(int)
    co_occur = (_Y.T @ _Y).astype(float)
    d = co_occur.shape[0]
    if normalize:
        diags = np.diagonal(co_occur)
        for i in range(d):
            for j in range(d):
                co_occur[i, j] = co_occur[i, j] / np.max([diags[i], diags[j]])
    if log:
        co_occur = np.log(co_occur)
    return co_occur
