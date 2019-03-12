import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import comb
from pyrolite.util.math import nancov
from pyrolite.comp.codata import alr, inverse_alr, close
from collections import defaultdict
import warnings
import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


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

    Note
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
    n_obs = Y.shape[0]
    pattern_ids = np.zeros(n_obs)
    D = Y.shape[1]
    miss = ~np.isfinite(Y)
    rowmiss = np.array(np.nonzero(~np.isfinite(np.sum(Y, axis=1)))[0])
    max_pats = comb((D - 1) * np.ones(D - 2), np.arange(D - 2) + 1).sum().astype(int)
    pattern_ids[rowmiss] = max_pats + 2  # initialise to high value
    pattern_dict = defaultdict(dict)

    pattern_no = 0  # 0 = no missing data
    pattern_dict[int(0)] = {
        "pattern": np.zeros(D).astype(bool),
        "freq": np.sum(pattern_ids == 0),
    }
    indexes = np.arange(n_obs).astype(int)
    indexes = indexes[pattern_ids[indexes] > pattern_no]  # only look at md rows
    for idx in indexes:
        if pattern_ids[idx] > pattern_no:  # has missing data
            pattern_no += 1
            pattern_ids[idx] = pattern_no
            pattern = miss[idx, :]
            pattern_dict[int(pattern_no)] = {"pattern": pattern, "freq": 0}
            if idx < n_obs:
                _rix = np.arange(idx + 1, n_obs)
                to_compare = _rix[pattern_ids[_rix] > pattern_no]
                where_same = to_compare[(miss[to_compare, :] == pattern).all(axis=1)]
                pattern_ids[where_same] = pattern_no
    for ID in np.unique(pattern_ids).astype(int):
        pattern_dict[ID]["freq"] = np.sum(pattern_ids == ID)
    return pattern_ids, pattern_dict


def _little_sweep(G, k: int = 1, verify=False):
    """
    Parameters
    ---------------
    G : :class:`numpy.ndarray`
        Input array to sweep.
    k : :class:`int`
        Index to sweep on.
    verify : :class:`bool`
        Whether to verify valid matrix input.

    References
    ----------
        Little R. J. A. and Rubin D. B. (2014).
        Statistical analysis with missing data. 2nd ed., Wiley, Hoboken, N.J.

    Note
    ------
        The sweep operator is defined for symmetric matrices as follows. A p x p
        symmetric matrix G is said to be swept on row and column k if it is replaced by
        another symmetric p x p matrix H with elements defined as follows:

        h_kk = -1 / g_kk
        h_jk = h_kj = g_jk / g_kk ; j != k
        h_jl = g_jl - g_jk * g_kl / g_kk ; j != k; l != k

    """
    H = np.asarray(G).copy()
    n = H.shape[0]
    if verify:
        if H.shape != (n, n):
            raise ValueError("Not a square array")
        if not np.allclose(H - H.T, 0):
            raise ValueError("Not a symmetrical array")
    inds = [i for i in np.linspace(0, n - 1, n).astype(int) if not i == k]
    H[k, k] = -1 / G[k, k]
    assert G[k, k] != 0.0
    H[k, inds] = -G[k, inds] * H[k, k]  # divide row k by D
    H[inds, k] = -G[inds, k] * H[k, k]  # divide column k by D
    for j in inds:
        for l in inds:
            H[j, l] = G[j, l] - H[j, k] * G[k, l]

    return H


def _reg_sweep(M: np.ndarray, C: np.ndarray, varobs: np.ndarray, error_threshold=200):
    """
    A function to cacluate estimated regression coefficients and residial covariance
    matrix for missing variables.
    After Palarea-Albaladejo and Martín-Fernández (2008) [#ref_1]_.

    Parameters
    -----------
    M : :class:`numpy.ndarray`
        Array of means.
    C : :class:`numpy.ndarray`
        Covariance matrix.
    varobs : :class:`numpy.ndarray`
        Boolean array indicating which variables are included in the regression model.

    Returns
    --------
    β : :class:`numpy.ndarray`
        Array of estimated means.
    σ2_res : :class:`numpy.ndarray`
        Residuals.

    References
    ----------
    .. [#ref_1] Palarea-Albaladejo J. and Martín-Fernández J. A. (2008)
            A modified EM alr-algorithm for replacing rounded zeros in compositional data sets.
            Computers & Geosciences 34, 902–917.
            doi: `10.1016/j.cageo.2007.09.015 <https://dx.doi.org/10.1016/j.cageo.2007.09.015>`__

    """
    assert np.isfinite(M).all()
    assert np.isfinite(C).all()
    assert (np.abs(np.log(M)) < error_threshold).all()
    p = M.shape[0]  # p > 0
    q = varobs.size  # q > 0
    i = np.ones(p)
    i[varobs] -= 1
    dep = np.array(np.nonzero(i))[0]  # indicies where i is nonzero
    # Shift the non-zero element to the end for pivoting
    reor = np.concatenate(([0], varobs + 1, dep + 1), axis=0)  #
    A = augmented_covariance_matrix(M, C)
    A = A[reor, :][:, reor]
    Astart = A.copy()
    assert (np.diag(A) != 0).all()  # Not introducing extra zeroes
    for n in range(q):  # for
        A = _little_sweep(A, n)
    if not np.isfinite(A).all():  # Typically caused by infs
        A[~np.isfinite(A)] = 0
    β = A[0 : q + 1, q + 1 : p + 1]
    σ2_res = A[q + 1 : p + 1, q + 1 : p + 1]
    return β, σ2_res


def random_composition_missing(size=1000, D=4, propnan=0.1, MAR=True):
    """
    Generate a simulated random compositional dataset with missing data.

    Parameters
    -----------
    size : :class:`int`
        Size of the dataset.
    D : :class:`int`
        Dimensionality of the dataset.
    propnan : :class:`float`, [0, 1)
        Proportion of missing values in the output dataset.
    MAR : :class:`bool`
        Whether to have missing at random data (:code:`True`, random points selected),
        or instead have missing not at random data (:code:`False`, high-pass filter
        or 'thresholded').

    Returns
    --------
    :class:`numpy.ndarray`
        Simulated dataset with missing values.
    """
    data = np.array(
        [np.ones(size) * 0.1 + np.random.rand(size) * 0.001]
        + [np.random.rand(size) for i in range(D - 1)]
    ).T
    data = close(data)

    if MAR:
        nnan = int(propnan * size)
        for _ in range(nnan):
            for i in range(data.shape[1] - 1):
                data[np.random.randint(size), i + 1] = np.nan
    else:
        thresholds = np.percentile(data, propnan * 100, axis=0)
        print(thresholds)
        data[:, 1:] = np.where(data[:, 1:] < np.tile(thresholds[1:], size).reshape(size, D-1), np.nan, data[:, 1:])
    return data


def EMCOMP(
    X,
    threshold=None,
    tol=0.0001,
    # tfm=lambda x, *args: ilr(x),
    # inv_tfm=lambda x, *args: inverse_ilr(x),
):
    """
    EMCOMP replaces rounded zeros in a compositional data set based on a set of
    thresholds. After Palarea-Albaladejo and Martín-Fernández (2008) [#ref_1]_.


    Parameters
    ----------
    X  : :class:`numpy.ndarray`
        Dataset with rounded zeros
    threshold : :class:`numpy.ndarray`
        Array of threshold values for each component as a proprotion.

    Returns
    --------
    X_est : :class:`numpy.ndarray`
        Dataset with rounded zeros replaced.
    prop_zeros : :class:`float`
       Proportion of zeros in the original data set.
    n_iters : :class:`int`
        Number of iterations needed for convergence.

    Note
    -----
        At least one component without missing values is needed for the divisor. Rounded zeros/
        missing values are replaced by values below their respective detection limits.

    References
    ----------
    .. [#ref_1] Palarea-Albaladejo J. and Martín-Fernández J. A. (2008)
            A modified EM alr-algorithm for replacing rounded zeros in compositional data sets.
            Computers & Geosciences 34, 902–917.
            doi: `10.1016/j.cageo.2007.09.015 <https://dx.doi.org/10.1016/j.cageo.2007.09.015>`__
    """
    X = X.copy()
    n_obs, p = X.shape
    D = p

    """Close the X rows to 1"""
    X = np.divide(X, np.nansum(X, axis=1)[:, np.newaxis])
    """Convert zeros into missing values"""
    X[X == 0] = np.nan
    """Use a divisor free of missing values"""
    assert np.all(np.isfinite(X), axis=0).any()
    pos = np.argmax(np.all(np.isfinite(X), axis=0))
    Yden = X[:, pos]
    """Compute the matrix of censure points Ψ"""
    cpoints = (
        np.ones((n_obs, 1)) @ np.log(threshold[np.newaxis, :])
        - np.log(Yden[:, np.newaxis]) @ np.ones((1, p))
        - np.spacing(1.0)  # Machine epsilon
    )

    assert np.isfinite(cpoints).all()
    cpoints = cpoints[:, [i for i in range(D) if not i == pos]]
    prop_zeroes = np.count_nonzero(~np.isfinite(X)) / (n_obs * p)
    Y = alr(X, pos)
    # Y = np.vstack((alr(X, pos).T, np.zeros(n_obs))).T
    # ---------------Log Space--------------------------------
    n_obs, p = Y.shape
    M = np.nanmean(Y, axis=0)  # μ0
    C = nancov(Y)  # Σ0
    assert np.isfinite(M).all() and np.isfinite(C).all()
    """
    ------------------------------------------------
    Stage 2: Find and enumerate missing data patterns.
    ------------------------------------------------
    """
    pattern_ids, pattern_dict = md_pattern(Y)
    """
    Regression against other variables ------------------------------------------
    """
    another_iter = True
    niters = 0
    while another_iter:
        niters += 1
        Mnew = M
        Cnew = C
        Ystar = Y.copy()
        V = np.zeros((p, p))
        for p_no in np.unique(pattern_ids):
            patternrows = np.arange(pattern_ids.size)[pattern_ids == p_no]
            ni = patternrows.size
            observed = np.isfinite(Y[patternrows[0], :])
            varobs, varmiss = (
                np.arange(D - 1)[~pattern_dict[p_no]["pattern"]],
                np.arange(D - 1)[pattern_dict[p_no]["pattern"]],
            )
            nvarobs, nvarmiss = varobs.size, varmiss.size
            sigmas = np.zeros(p)

            if nvarobs:  # Only where there observations
                B, σ2_res = _reg_sweep(Mnew, Cnew, varobs)
                if B.size:  # If there are missing elements in the pattern
                    B = B.flatten()
                    # estimate mising y
                    Ystar[np.ix_(patternrows, varmiss)] = (
                        np.ones(ni) * B[0]
                        + Y[np.ix_(patternrows, varobs)] @ B[1 : (nvarobs + 1)]
                    )[:, np.newaxis]

                    # assert np.isfinite(Ystar[np.ix_(patternrows, varmiss)]).all()

                    sigmas[varmiss] = np.sqrt(np.diag(σ2_res))
                    _diff = (
                        cpoints[np.ix_(patternrows, varmiss)]
                        - Ystar[np.ix_(patternrows, varmiss)]
                    )
                    _std_devs = np.divide(_diff, sigmas[varmiss])
                    _ϕ = stats.norm.pdf(_std_devs, loc=0, scale=1)
                    _Φ = stats.norm.cdf(_std_devs, loc=0, scale=1)
                    _sigdevs = sigmas[varmiss] * _ϕ / _Φ
                    _sigdevs = np.where(np.isfinite(_sigdevs), _sigdevs, 0.0)
                    Ystar[np.ix_(patternrows, varmiss)] -= _sigdevs
                    V[np.ix_(varmiss, varmiss)] += σ2_res * ni

        """Update and store parameter vector (μ(t), Σ(t))."""
        M = np.nanmean(Ystar, axis=0)
        dif = Ystar - np.ones(n_obs)[:, np.newaxis] @ M[np.newaxis, :]
        dif[np.isnan(dif)] = 0.0
        try:
            PearsonCorr = np.dot(dif.T, dif)
            assert n_obs > 1
            C = (PearsonCorr + V) / (n_obs - 1)
            assert np.isfinite(C).all()
        except AssertionError:
            print(PearsonCorr)
            C = Cnew
        Mdif, Cdif = np.nanmax(np.abs(M - Mnew)), np.nanmax(np.abs(C - Cnew))
        if np.nanmax(np.vstack([Mdif, Cdif]).flatten()) < tol:  # Convergence checking
            another_iter = False

    Xstar = inverse_alr(Ystar, pos)
    return Xstar, prop_zeroes, niters


def impute_ratios(ratios: pd.DataFrame):
    """
    Imputation function utilizing pandas which is used to fill out the
    aggregated ratio matrix via chained ratio multiplication akin to
    internal standardisation (e.g. Ti / MgO = Ti/SiO2 * SiO2 / MgO).

    Parameters
    ---------------
    ratios : :class:`pandas.DataFrame`
        Dataframe of ratios to impute.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame of imputed ratios.
    """
    with warnings.catch_warnings():
        # can get empty arrays which raise RuntimeWarnings
        # consider changing to np.errstate
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for IS in ratios.columns:
            ser = ratios.loc[:, IS]
            if ser.isnull().any():
                non_null_idxs = ser.loc[~ser.isnull()].index.values
                null_idxs = ser.loc[ser.isnull()].index.values
                for null in null_idxs:  # e.g.  Ti / MgO = Ti/SiO2 * SiO2 / MgO
                    # e.g. SiO2/MgO ratios
                    inverse_ratios = ratios.loc[null, non_null_idxs]
                    # e.g. Ti/SiO2 ratios
                    non_null_ISratios = ratios.loc[non_null_idxs, IS]
                    predicted_ratios = inverse_ratios * non_null_ISratios
                    ratios.loc[null, IS] = np.exp(np.nanmean(np.log(predicted_ratios)))
    return ratios


def np_impute_ratios(ratios: np.ndarray):
    """
    Imputation function utilizing numpy which is used to fill out the
    aggregated ratio matrix via chained ratio multiplication akin to
    internal standardisation (e.g. Ti / MgO = Ti/SiO2 * SiO2 / MgO).

    Parameters
    ---------------
    ratios : :class:`numpy.ndarray`
        Array of ratios to impute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of imputed ratios.
    """
    finite = np.isfinite(ratios)
    not_finite = ~finite
    if not_finite.any():
        where_not_finite = np.argwhere(not_finite)
        _ixs, _iys = where_not_finite.T
        ixs = _ixs[~(_ixs == _iys)]
        iys = _iys[~(_ixs == _iys)]
        where_not_finite = np.stack((ixs, iys)).T
        excludes = np.empty((ixs.size, ratios.shape[0] - 2)).astype(int)
        indicies = np.arange(ratios.shape[0]).astype(int)
        for enm_ix in np.arange(ixs.size):
            excludes[enm_ix] = np.setdiff1d(indicies, where_not_finite[enm_ix])

        for enm_ix in np.arange(ixs.size):
            ex = excludes[enm_ix]
            ix, iy = where_not_finite[enm_ix].T
            with warnings.catch_warnings():
                # can get empty arrays which raise RuntimeWarnings
                # consider changing to np.errstate
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ratios[ix, iy] = np.nanmean(ratios[ix, ex] + ratios[ex, iy])
    return ratios
