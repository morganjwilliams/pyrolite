import numpy as np
import logging
from scipy.stats.kde import gaussian_kde


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def sample_kde(data, *coords, renorm=False):
    """
    Sample a Kernel Density Estimate based on data,
    at points or a grid defined by coords: xi, yi.

    Parameters
    ------------
    data : :class:`numpy.ndarray`
        Source data to estimate the kernel density estimate.
    coords : :class:`numpy.ndarray`
        Arrays of coordiantes (xi, yi,..) to sample the KDE estimate at.

    Returns
    ----------
    :class:`numpy.ndarray`
    """
    data = data[:, np.isfinite(data).all(axis=0)]
    k = gaussian_kde(data)
    kcoords = np.vstack([c.flatten() for c in coords])
    zi = k(kcoords).T
    zi = zi.reshape(coords[0].shape)
    if renorm:
        zi = zi / np.nanmax(zi)
    return zi


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
