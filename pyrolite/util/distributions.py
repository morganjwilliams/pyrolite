import numpy as np
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def lognorm_to_norm(mu, sigma):
    """
    Calculate mean and variance for a normal random variable from the lognormal
    parameters :code:`mu` and :code:`sigma`.

    Parameters
    -----------
    mu : :class:`float`
        Parameter :code:`mu` for the lognormal distribution.
    sigma : :class:`float`
        :code:`sigma` for the lognormal distribution.

    Returns
    --------
    mean : :class:`float`
        Mean of the normal distribution.
    sigma : :class:`float`
        Variance of the normal distribution.
    """
    return (
        np.exp(mu + 0.5 * sigma ** 2),
        np.sqrt(np.exp(2 * mu + sigma ** 2) * (np.exp(sigma ** 2) - 1)),
    )


def norm_to_lognorm(mean, sigma):
    """
    Calculate :code:`mu` and :code:`sigma` parameters for a lognormal random variable
    with a given mean and variance.

    Parameters
    -----------
    mean : :class:`float`
        Mean of the normal distribution.
    sigma : :class:`float`
        :code:`sigma` of the normal distribution.

    Returns
    --------
    mu : :class:`float`
        Parameter :code:`mu` for the lognormal distribution.
    sigma : :class:`float`
        :code:`sigma` of the lognormal distribution.
    """
    return (
        2 * np.log(mean) - 0.5 * np.log(sigma ** 2 + mean ** 2),
        np.sqrt(-2 * np.log(mean) + np.log(sigma ** 2 + mean ** 2)),
    )
