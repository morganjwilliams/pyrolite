import numpy as np
from scipy.special import comb
from collections import defaultdict

def md_pattern(Y):
    """
    Get the missing data patterns from an array.

    Parameters
    ------------
    Y : :class:`numpy.ndarray`
        Input dataset.

    Returns
    ---------
    pattern_ids : :class:`numpy.ndarray`
        Pattern ID array.
    pattern_dict : :class:`dict`
        Dictionary of patterns indexed by pattern IDs. Contains a pattern and count
        for each pattern ID.
    """
    N, D = Y.shape
    pID = np.zeros(N)
    Ymiss = ~np.isfinite(Y)
    rows = np.arange(N)[~np.isfinite(np.sum(Y, axis=1))]
    max_pats = comb((D - 1) * np.ones(D - 2), np.arange(D - 2) + 1).sum().astype(int)
    pID[rows] = max_pats + 2  # initialise to high value
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
