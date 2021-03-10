import numpy as np
import scipy.stats as stats
from pyrolite.util.math import nancov, augmented_covariance_matrix
from pyrolite.util.missing import md_pattern
from pyrolite.comp.codata import ALR, inverse_ALR, close
from ..util.log import Handle

logger = Handle(__name__)


def _little_sweep(G, k: int = 0, verify=False):
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

    Notes
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
        if not np.isfinite(H).all():
            raise ValueError("Not all finite.")
        if not np.allclose(H - H.T, 0):
            raise ValueError("Not a symmetrical array")
    inds = [i for i in np.linspace(0, n - 1, n).astype(int) if not np.isclose(i, k)]
    assert np.isfinite(G[k, k]) & (G[k, k] != 0.0)
    H[k, k] = -1 / G[k, k]
    H[k, inds] = -G[k, inds] * H[k, k]  # divide row k by D
    H[inds, k] = -G[inds, k] * H[k, k]  # divide column k by D
    for j in inds:
        for l in inds:
            H[j, l] = G[j, l] - H[j, k] * G[k, l]

    return H


def _multisweep(G, ks):
    """
    Sweep G along all indexes ks.

    Parameters
    -----------
    G : :class:`numpy.ndarray`
        Augmented covariance matrix to sweep.
    ks : :class:`numpy.ndarray`
        Indicies to sweep.

    Returns
    --------
    :class:`numpy.ndarray`
    """
    H = G.copy()
    for k in ks:
        H = _little_sweep(H, k=k)
    return H


def _reg_sweep(M: np.ndarray, C: np.ndarray, varobs: np.ndarray, error_threshold=None):
    """
    Performs multiple sweeps of the augmented covariance matrix and extracts the
    regression coefficients :math:`\beta_{0} \cdots \beta_(d)` and residial covariance
    for the regression of missing variables against observed variables for a given
    missing data pattern. Translated from matlab to python from Palarea-Albaladejo
    and Martín-Fernández (2008) [#ref_1]_. Note that this algorithm requires at least
    two columns free of missing values.

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
        Array of estimated regression coefficients.
    σ2_res : :class:`numpy.ndarray`
        Residuals.

    References
    ----------
    .. [#ref_1] Palarea-Albaladejo J. and Martín-Fernández J. A. (2008)
            A modified EM ALR-algorithm for replacing rounded zeros in compositional data sets.
            Computers & Geosciences 34, 902–917.
            doi: `10.1016/j.cageo.2007.09.015 <https://dx.doi.org/10.1016/j.cageo.2007.09.015>`__

    """
    assert np.isfinite(M).all()
    assert np.isfinite(C).all()
    if error_threshold is not None:
        assert (np.abs(M) < error_threshold).all()  # avoid runaway expansion
    dimension = M.size  # p > 0
    nvarobs = varobs.size  # q > 0 # number of observed variables
    dep = np.array([i for i in np.arange(dimension) if not i in varobs])
    # Shift the non-zero element to the end for pivoting
    reor = np.concatenate(([0], varobs + 1, dep + 1), axis=0)  #
    A = augmented_covariance_matrix(M, C)
    A = A[reor, :][:, reor]
    # Astart = A.copy(deep=True)
    assert (np.diag(A) != 0).all()  # Not introducing extra zeroes
    A = _multisweep(A, range(nvarobs + 1))
    """
    A is of form:
    -D  | E
    E.T | F
    """
    # if not np.isfinite(A).all():  # Typically caused by infs
    #    A[~np.isfinite(A)] = 0
    assert np.isfinite(A).all()
    β = A[0 : nvarobs + 1, nvarobs + 1 : dimension + 1]
    σ2_res = A[nvarobs + 1 :, nvarobs + 1 :]
    return β, σ2_res


def EMCOMP(
    X,
    threshold=None,
    tol=0.0001,
    convergence_metric=lambda A, B, t: np.linalg.norm(np.abs(A - B)) < t,
    max_iter=30,
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
    tol : :class:`float`
        Tolerance to check for convergence.
    convergence_metric : :class:`callable`
        Callable function to check for convergence. Here we use a compositional distance
        rather than a maximum absolute difference, with very similar performance.
        Function needs to accept two :class:`numpy.ndarray` arguments and third
        tolerance argument.
    max_iter : :class:`int`
        Maximum number of iterations before an error is thrown.

    Returns
    --------
    X_est : :class:`numpy.ndarray`
        Dataset with rounded zeros replaced.
    prop_zeros : :class:`float`
       Proportion of zeros in the original data set.
    n_iters : :class:`int`
        Number of iterations needed for convergence.

    Notes
    -----

        * At least one component without missing values is needed for the divisor.
          Rounded zeros/missing values are replaced by values below their respective
          detection limits.

        * This routine is not completely numerically stable as written.

    Todo
    -------
        * Implement methods to deal with variable decection limits (i.e thresholds are array shape :code:`(N, D)`)
        * Conisder non-normal models for data distributions.
        * Improve numerical stability to reduce the chance of :code:`np.inf` appearing.

    References
    ----------
    .. [#ref_1] Palarea-Albaladejo J. and Martín-Fernández J. A. (2008)
            A modified EM ALR-algorithm for replacing rounded zeros in compositional data sets.
            Computers & Geosciences 34, 902–917.
            doi: `10.1016/j.cageo.2007.09.015 <https://dx.doi.org/10.1016/j.cageo.2007.09.015>`__

    """
    X = X.copy()
    n_obs, D = X.shape
    X = close(X, sumf=np.nansum)
    # ---------------------------------
    # Convert zeros into missing values
    # ---------------------------------
    X = np.where(np.isclose(X, 0.0), np.nan, X)
    # Use a divisor free of missing values
    assert np.isfinite(X).all(axis=0).any()
    pos = np.argmax(np.isfinite(X).all(axis=0))
    Yden = X[:, pos]
    # --------------------------------------
    # Compute the matrix of censure points Ψ
    # --------------------------------------
    # need an equivalent concept for ilr
    cpoints = (
        np.ones((n_obs, 1)) @ np.log(threshold[np.newaxis, :])
        - np.log(Yden[:, np.newaxis]) @ np.ones((1, D))
        - np.spacing(1.0)  # Machine epsilon
    )
    assert np.isfinite(cpoints).all()
    cpoints = cpoints[:, [i for i in range(D) if not i == pos]]  # censure points
    prop_zeroes = np.count_nonzero(~np.isfinite(X)) / (n_obs * D)
    Y = ALR(X, pos)
    # ---------------Log Space--------------------------------
    LD = Y.shape[1]
    M = np.nanmean(Y, axis=0)  # μ0
    C = nancov(Y)  # Σ0
    assert np.isfinite(M).all() and np.isfinite(C).all()

    # --------------------------------------------------
    # Stage 2: Find and enumerate missing data patterns
    # --------------------------------------------------
    pID, pD = md_pattern(Y)
    # -------------------------------------------
    # Stage 3: Regression against other variables
    # -------------------------------------------
    logger.debug(
        "Starting Iterative Regression for Matrix : ({}, {})".format(n_obs, LD)
    )
    another_iter = True
    niters = 0
    while another_iter:
        niters += 1
        Mnew, Cnew = M.copy(), C.copy()
        Ystar = Y.copy()
        V = np.zeros((LD, LD))

        for p_no in np.unique(pID):
            logger.debug("Pattern ID: {}, {}".format(p_no, pD[p_no]["pattern"]))
            rows = np.arange(pID.size)[pID == p_no]  # rows with this pattern
            varobs, varmiss = (
                np.arange(D - 1)[~pD[p_no]["pattern"]],
                np.arange(D - 1)[pD[p_no]["pattern"]],
            )
            sigmas = np.zeros((LD))
            assert np.isfinite(Y[np.ix_(rows, varobs)]).all()
            assert (~np.isfinite(Y[np.ix_(rows, varmiss)])).all()
            if varobs.size and varmiss.size:  # Non-completely missing, but missing some
                logger.debug(
                    "Regressing {} rows for pattern {} | {}.".format(
                        rows.size,
                        "".join(varobs.astype(str)),
                        "".join(varmiss.astype(str)),
                    )
                )
                B, σ2_res = _reg_sweep(Mnew, Cnew, varobs)
                assert B.shape == (varobs.size + 1, varmiss.size)
                assert σ2_res.shape == (varmiss.size, varmiss.size)
                assert np.isfinite(B).all()
                logger.debug(
                    "Current Estimator (1, {})".format(
                        ", ".join(["β{}".format(i) for i in range(B.shape[0] - 1)])
                    )
                )

                Ystar[np.ix_(rows, varmiss)] = np.ones((rows.size, 1)) * B[0, :] + (
                    (Y[np.ix_(rows, varobs)] @ B[1 : (varobs.size + 1), :])
                )
                sigmas[varmiss] = np.sqrt(np.diag(σ2_res))
                assert np.isfinite(sigmas[varmiss]).all()

                x = (  # position of threshold values relative to estimated means
                    cpoints[np.ix_(rows, varmiss)] - Ystar[np.ix_(rows, varmiss)]
                )
                x /= sigmas[varmiss][np.newaxis, :]  # as standard deviations
                assert np.isfinite(x).all()
                # ----------------------------------------------------
                # Calculate inverse Mills Ratio for Heckman correction
                # ----------------------------------------------------
                ϕ = stats.norm.pdf(x, loc=0, scale=1)  # pdf
                Φ = stats.norm.cdf(x, loc=0, scale=1)  # cdf
                Φ[np.isclose(Φ, 0)] = np.finfo(np.float).eps * 2
                assert (Φ > 0).all()  # if its not, infinity will be introduced
                inversemills = ϕ / Φ
                Ystar[np.ix_(rows, varmiss)] = (
                    Ystar[np.ix_(rows, varmiss)] - sigmas[varmiss] * inversemills
                )
                V[np.ix_(varmiss, varmiss)] += σ2_res * rows.size
        assert np.isfinite(V).all()
        # -----------------------------------------------
        # Update and store parameter vector (μ(t), Σ(t)).
        # -----------------------------------------------
        logger.debug("Regression finished.")
        M = np.nanmean(Ystar, axis=0)
        Ydevs = Ystar - np.ones((n_obs, 1)) * M
        Ydevs[~np.isfinite(Ydevs)] = 0.0  # remove nonfinite components
        PC = np.dot(Ydevs.T, Ydevs)
        logger.debug("Correlation:\n{}".format(PC / (n_obs - 1)))
        C = (PC + V) / (n_obs - 1)

        logger.debug("Average diff: {}".format(np.mean(Ydevs, axis=0)))
        assert np.isfinite(C).all()
        # --------------------
        # Convergence checking
        # --------------------
        if convergence_metric(M, Mnew, tol) & convergence_metric(C, Cnew, tol):
            another_iter = False
            logger.debug("Convergence achieved.")

        another_iter = another_iter & (niters < max_iter)
        logger.debug("Iterations Continuing: {}".format(another_iter))
    #----------------------------
    # Back to compositional space
    # ---------------------------
    logger.debug("Finished. Inverting to compositional space.")
    Xstar = inverse_ALR(Ystar, pos)
    return Xstar, prop_zeroes, niters
