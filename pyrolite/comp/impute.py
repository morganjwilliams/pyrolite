import numpy as np
import pandas as pd
import scipy.stats as stats
from pyrolite.util.math import nancov, augmented_covariance_matrix
from pyrolite.util.missing import md_pattern
from pyrolite.comp.codata import alr, inverse_alr, ilr, inverse_ilr, close
import warnings
import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


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

    Returns
    --------
    H : :class:`numpy.ndarray`
        Swept array.

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
    assert np.isfinite(G[k, k]) & (G[k, k] != 0.0)
    H[k, k] = -1 / G[k, k]
    H[k, inds] = -G[k, inds] * H[k, k]  # divide row k by D
    H[inds, k] = -G[inds, k] * H[k, k]  # divide column k by D
    for j in inds:
        for l in inds:
            H[j, l] = G[j, l] - H[j, k] * G[k, l]

    return H


def _reg_sweep(M: np.ndarray, C: np.ndarray, varobs: np.ndarray, error_threshold=None):
    """
    A function to cacluate estimated regression coefficients and residial covariance
    matrix for missing variables.
    After Palarea-Albaladejo and Martín-Fernández (2008) [#ref_1]_.

    Parameters
    -----------
    M : :class:`numpy.ndarray`
        Array of means of shape :code:`(D, )`.
    C : :class:`numpy.ndarray`
        Covariance of shape :code:`(D, D)`.
    varobs : :class:`numpy.ndarray`
        Boolean array indicating which variables are included in the regression model,
        of shape :code:`(D, )`
    error_threshold : :class:`float`
        Low-pass threshold at which an error will result, of shape :code:`(D, )`.
        Effectively limiting mean values to :math:`e^{threshold}`.

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
    if error_threshold is not None:
        assert (np.abs(M) < error_threshold).all()
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


def EMCOMP(
    X,
    threshold=None,
    tol=0.0001,
    convergence_metric=lambda A, B, t: np.linalg.norm(np.abs(A - B)) < t,
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
    convergence_metric : :class:`callable`
        Callable function to check for convergence. Here we use a compositional distance
        rather than a maximum absolute difference, with very similar performance.

    Notes
    -----
        * At least one component without missing values is needed for the divisor. Rounded zeros/
            missing values are replaced by values below their respective detection limits.

        * This routine is not completely numerically stable as written.

    Todo
    -------
        * Implement methods to deal with variable decection limits (i.e thresholds are array shape :code`(N, D)`)
        * Conisder non-normal models for data distributions.
        * Improve numerical stability to reduce the chance of :code:`np.inf` appearing.

    References
    ----------
    .. [#ref_1] Palarea-Albaladejo J. and Martín-Fernández J. A. (2008)
            A modified EM alr-algorithm for replacing rounded zeros in compositional data sets.
            Computers & Geosciences 34, 902–917.
            doi: `10.1016/j.cageo.2007.09.015 <https://dx.doi.org/10.1016/j.cageo.2007.09.015>`__
    """
    X = X.copy()
    n_obs, D = X.shape
    X = close(X, sumf=np.nansum)
    """Convert zeros into missing values"""
    X = np.where(np.isclose(X, 0.0), np.nan, X)
    """Use a divisor free of missing values"""
    assert np.isfinite(X).all(axis=0).any()
    pos = np.argmax(np.isfinite(X).all(axis=0))
    Yden = X[:, pos]
    """Compute the matrix of censure points Ψ"""
    # need an equivalent concept for ilr
    cpoints = (
        np.ones((n_obs, 1)) @ np.log(threshold[np.newaxis, :])
        - np.log(Yden[:, np.newaxis]) @ np.ones((1, D))
        - np.spacing(1.0)  # Machine epsilon
    )
    assert np.isfinite(cpoints).all()
    cpoints = cpoints[:, [i for i in range(D) if not i == pos]]
    prop_zeroes = np.count_nonzero(~np.isfinite(X)) / (n_obs * D)
    Y = alr(X, pos)
    # ---------------Log Space--------------------------------
    LD = Y.shape[1]
    M = np.nanmean(Y, axis=0)  # μ0
    C = nancov(Y)  # Σ0
    assert np.isfinite(M).all() and np.isfinite(C).all()
    """
    ------------------------------------------------------------------------------------
    Stage 2: Find and enumerate missing data patterns.
    ------------------------------------------------------------------------------------
    """
    pID, pD = md_pattern(Y)
    """
    ------------------------------------------------------------------------------------
    Stage 3: Regression against other variables
    ------------------------------------------------------------------------------------
    """
    another_iter = True
    niters = 0
    while another_iter:
        niters += 1
        Mnew, Cnew = M, C
        Ystar = Y.copy()
        V = np.zeros((LD, LD))

        for p_no in np.unique(pID):
            rows = np.arange(pID.size)[pID == p_no]  # rows with this pattern
            ni = rows.size  # number of rows
            varobs, varmiss = (
                np.arange(D - 1)[~pD[p_no]["pattern"]],
                np.arange(D - 1)[pD[p_no]["pattern"]],
            )

            sigmas = np.zeros(LD)

            if varobs.size and varmiss.size:  # Non-completely missing, but missing some
                B, σ2_res = _reg_sweep(Mnew, Cnew, varobs)

                B = B.flatten()
                Ystar[np.ix_(rows, varmiss)] = (
                    np.ones(ni) * B[0]
                    + Y[np.ix_(rows, varobs)] @ B[1 : (varobs.size + 1)]
                )[
                    :, np.newaxis
                ]  # regression

                V[np.ix_(varmiss, varmiss)] += σ2_res * rows.size
                sigmas[varmiss] = np.sqrt(np.diag(σ2_res))
                diff = (  # distance below thresholds
                    cpoints[np.ix_(rows, varmiss)]  # threshold values
                    - Ystar[np.ix_(rows, varmiss)]
                )
                SD = diff / sigmas[varmiss][np.newaxis, :]  # standard deviations
                ϕ = stats.norm.pdf(SD, loc=0, scale=1)  # pdf
                Φ = stats.norm.cdf(SD, loc=0, scale=1)  # cdf
                ds = sigmas[varmiss] * ϕ / Φ
                ds = np.where(np.isfinite(ds), ds, 0.0)
                Ystar[np.ix_(rows, varmiss)] -= ds

        """Update and store parameter vector (μ(t), Σ(t))."""
        M = np.nanmean(Ystar, axis=0)

        dif = Ystar - np.ones(n_obs)[:, np.newaxis] @ M[np.newaxis, :]
        dif[np.isnan(dif)] = 0.0
        PearsonCorr = np.dot(dif.T, dif)
        C = (PearsonCorr + V) / (n_obs - 1)
        try:
            assert np.isfinite(C).all()
        except AssertionError:
            logger.warning("Covariance matrix not finite.")
            C = Cnew

        # Convergence checking
        if convergence_metric(M, Mnew, tol) & convergence_metric(C, Cnew, tol):
            another_iter = False

    # Back to compositional space
    Xstar = inverse_alr(Ystar, pos)
    return Xstar, prop_zeroes, niters


def impute_ratios(ratios: pd.DataFrame):
    """
    Imputation function utilizing pandas which is used to fill out the
    aggregated ratio matrix via chained ratio multiplication akin to
    internal standardisation (e.g. Ti / MgO = Ti/SiO2 * SiO2 / MgO).

    .. warning:: Not used in the wild.

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

    .. warning:: Not used in the wild.

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
