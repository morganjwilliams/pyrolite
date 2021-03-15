"""
Functions for transforming ionic radii to and from atomic number for the visualisation
of REE patterns.
"""
import numpy as np
from ...geochem.ind import get_ionic_radii, REE
from ..log import Handle

logger = Handle(__name__)


def REE_z_to_radii(z, fit=None, degree=7, **kwargs):
    """
    Estimate the ionic radii which would be approximated by a given atomic number
    based on a provided (or calcuated) fit for the Rare Earth Elements.

    Parameters
    ----------
    z : :class:`float` | :class:`list` | :class:`numpy.ndarray`
        Atomic nubmers to be converted.
    fit : callable
        Callable function optionally specified; if not specified it will be calculated
        from Shannon Radii.
    degree : :class:`int`
        Degree of the polynomial fit between atomic number and radii.

    Returns
    -------
    r : :class:`float` | :class:`numpy.ndarray`
        Approximate atomic nubmers for given radii.
    """
    if fit is None:
        radii = np.array(
            get_ionic_radii(REE(dropPm=False), charge=3, coordination=8, **kwargs)
        )
        p, resids, rank, s, rcond = np.polyfit(
            np.arange(57, 72), radii, degree, full=True
        )

        def fit(x):
            return np.polyval(p, x)

    r = fit(z)
    return r


def REE_radii_to_z(r, fit=None, degree=7, **kwargs):
    """
    Estimate the atomic number which would be approximated by a given ionic radii
    based on a provided (or calcuated) fit for the Rare Earth Elements.

    Parameters
    ----------
    r : :class:`float` | :class:`list` | :class:`numpy.ndarray`
        Radii to be converted.
    fit : callable
        Callable function optionally specified; if not specified it will be calculated
        from Shannon Radii.
    degree : :class:`int`
        Degree of the polynomial fit between radii and atomic number.

    Returns
    -------
    z : :class:`float` | :class:`numpy.ndarray`
        Approximate atomic numbers for given radii.
    """
    if fit is None:
        radii = np.array(
            get_ionic_radii(REE(dropPm=False), charge=3, coordination=8, **kwargs)
        )
        p, resids, rank, s, rcond = np.polyfit(
            radii, np.arange(57, 72), degree, full=True
        )

        def fit(x):
            return np.polyval(p, x)

    z = fit(r)
    return z
