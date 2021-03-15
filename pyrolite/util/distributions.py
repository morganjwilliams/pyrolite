import numpy as np
import scipy.stats
from functools import partial
from ..util.math import flattengrid
from ..comp.codata import ILR, close
from .log import Handle

logger = Handle(__name__)


def get_scaler(*fs):
    """
    Generate a function which will transform columns of an array
    based on input functions (e.g. :code:`np.log` will log-transform the x values,
    :code:`None, np.log` will log-transform the y values but not the x).

    Parameters
    ------------
    fs
        A series of functions to apply to subsequent axes of an array.
    """

    def scaler(arr, fs=fs):
        A = arr.copy()
        for ix, f in enumerate(fs):
            if f is not None:
                A[:, ix] = f(A[:, ix])
        return A

    return partial(scaler, fs=fs)


def sample_kde(data, samples, renorm=False, transform=lambda x: x, bw_method=None):
    """
    Sample a Kernel Density Estimate at points or a grid defined.

    Parameters
    ------------
    data : :class:`numpy.ndarray`
        Source data to estimate the kernel density estimate; observations should be
        in rows (:code:`npoints, ndim`).
    samples : :class:`numpy.ndarray`
        Coordinates to sample the KDE estimate at (:code:`npoints, ndim`).
    transform
        Transformation used prior to kernel density estimate.
    bw_method : :class:`str`, :class:`float`, callable
        Method used to calculate the estimator bandwidth.
        See :func:`scipy.stats.kde.gaussian_kde`.

    Returns
    ----------
    :class:`numpy.ndarray`
    """
    # check shape info first
    data = np.atleast_2d(data)
    if data.shape[0] == 1:  # single row which should be a column
        logger.debug("Transposing data row to column format for KDE.")
        data = data.T

    tdata = transform(data)
    tdata = tdata[np.isfinite(tdata).all(axis=1), :]  # filter rows with nans

    K = scipy.stats.gaussian_kde(tdata.T, bw_method=bw_method)

    if isinstance(samples, list) and isinstance(samples[0], np.ndarray):  # meshgrid
        logger.debug("Sampling with meshgrid.")
        zshape = samples[0].shape
        ksamples = transform(flattengrid(samples))
    else:
        zshape = samples.shape[0]
        ksamples = transform(samples)

    # ensures shape is fine even if row is passed
    ksamples = ksamples.reshape(-1, tdata.shape[1])

    # samples shouldnt typically contain nans
    # ksamples = ksamples[np.isfinite(ksamples).all(axis=1), :]
    if not tdata.shape[1] == ksamples.shape[1]:
        logger.warn("Dimensions of data and samples do not match.")

    zi = K(ksamples.T)
    zi = zi.reshape(zshape)
    if renorm:
        logger.debug("Normalising KDE sample.")
        zi = zi / np.nanmax(zi)
    return zi


def sample_ternary_kde(data, samples, transform=ILR):
    """
    Sample a Kernel Density Estimate in ternary space points or a grid defined by
    samples.

    Parameters
    ------------
    data : :class:`numpy.ndarray`
        Source data to estimate the kernel density estimate (:code:`npoints, ndim`).
    samples : :class:`numpy.ndarray`
        Coordinates to sample the KDE estimate at  (:code:`npoints, ndim`)..
    transform
        Log-transformation used prior to kernel density estimate.

    Returns
    ----------
    :class:`numpy.ndarray`
    """
    return sample_kde(data, samples, transform=lambda x: transform(close(x)))


def lognorm_to_norm(mu, s):
    """
    Calculate mean and variance for a normal random variable from the lognormal
    parameters :code:`mu` and :code:`s`.

    Parameters
    -----------
    mu : :class:`float`
        Parameter :code:`mu` for the lognormal distribution.
    s : :class:`float`
        :code:`sigma` for the lognormal distribution.

    Returns
    --------
    mean : :class:`float`
        Mean of the normal distribution.
    sigma : :class:`float`
        Variance of the normal distribution.
    """
    mean = np.exp(mu + 0.5 * s ** 2)
    variance = (np.exp(s ** 2) - 1) * np.exp(2 * mu + s ** 2)
    return mean, np.sqrt(variance)


def norm_to_lognorm(mean, sigma, exp=True):
    """
    Calculate :code:`mu` and :code:`sigma` parameters for a lognormal random variable
    with a given mean and variance. Lognormal with parameters
    :code:`mean` and :code:`sigma`.

    Parameters
    -----------
    mean : :class:`float`
        Mean of the normal distribution.
    sigma : :class:`float`
        :code:`sigma` of the normal distribution.
    exp : :class:`bool`
        If using the :mod:`scipy.stats` parameterisation; this uses
        :code:`scale = np.exp(mu)`.

    Returns
    --------
    mu : :class:`float`
        Parameter :code:`mu` for the lognormal distribution.
    s : :class:`float`
        :code:`sigma` of the lognormal distribution.
    """
    mu = np.log(mean / np.sqrt(1 + sigma ** 2 / (mean ** 2)))
    v = np.log(1 + sigma ** 2 / (mean ** 2))
    if exp:  # scipy parameterisation of lognormal uses scale = np.exp(mu) !
        mu = np.exp(mu)
    return mu, np.sqrt(v)
