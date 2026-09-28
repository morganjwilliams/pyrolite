"""
Transformation utilites for matplotlib.
"""

from collections.abc import Callable

import numpy as np

from ...comp.codata import close
from ..log import Handle

logger = Handle(__name__)


def affine_transform(
    mtx: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
) -> Callable:
    """
    Construct a function which will perform a 2D affine transform based on
    a 3x3 affine matrix.

    Parameters
    -----------
    mtx : numpy.ndarray
    """

    def tfm(data):
        xy = data[:, :2]
        return (mtx @ np.vstack((xy.T[:2], np.ones(xy.T.shape[1]))))[:2]

    return tfm


def tlr_to_xy(tlr: np.ndarray) -> np.ndarray:
    """
    Transform a ternary coordinate system (top-left-right) to an xy-cartesian
    coordinate system.

    Parameters
    ----------
    tlr : numpy.ndarray
        Array of shape (n, 3) in the t-l-r coordinate system.

    Returns
    --------
    xy : numpy.ndarray
        Array of shape (n, 2) in the x-y coordinate system.
    """
    shear = affine_transform(np.array([[1, 1 / 2, 0], [0, 1, 0], [0, 0, 1]]))
    return shear(close(np.array(tlr)[:, [2, 0, 1]])).T


def xy_to_tlr(xy: np.ndarray) -> np.ndarray:
    """

    Parameters
    -----------
    xy : numpy.ndarray
        Array of shape (n, 2) in the x-y coordinate system.

    Returns
    --------
    tlr : numpy.ndarray
        Array of shape (n, 3) in the t-l-r coordinate system.
    """
    shear = affine_transform(np.array([[1, -1 / 2, 0], [0, 1, 0], [0, 0, 1]]))
    r, t = shear(xy)
    l = 1.0 - (r + t)
    return np.vstack([t, l, r]).T


def ABC_to_xy(ABC: np.ndarray, xscale: float = 1.0, yscale: float = 1.0) -> np.ndarray:
    """
    Convert ternary compositional coordiantes to x-y coordinates
    for visualisation within a triangle.

    Parameters
    -----------
    ABC : numpy.ndarray
        Ternary array (`samples, 3`).
    xscale : float
        Scale for x-axis.
    yscale : float
        Scale for y-axis.

    Returns
    --------
    numpy.ndarray
        Array of x-y coordinates (`samples, 2`)
    """
    assert ABC.shape[-1] == 3
    # transform from ternary to xy cartesian
    scale = affine_transform(np.array([[xscale, 0, 0], [0, yscale, 0], [0, 0, 1]]))
    shear = affine_transform(np.array([[1, 1 / 2, 0], [0, 1, 0], [0, 0, 1]]))
    xy = scale(shear(close(ABC)).T)
    return xy.T


def xy_to_ABC(xy: np.ndarray, xscale: float = 1.0, yscalel: float = 1.0) -> np.ndarray:
    """
    Convert x-y coordinates within a triangle to compositional ternary coordinates.

    Parameters
    -----------
    xy : numpy.ndarray
        XY array (`samples, 2`).
    xscale : float
        Scale for x-axis.
    yscale : float
        Scale for y-axis.

    Returns
    --------
    numpy.ndarray
        Array of ternary coordinates (`samples, 3`)
    """
    assert xy.shape[-1] == 2
    # transform from xy cartesian to ternary
    scale = affine_transform(
        np.array([[1 / xscale, 0, 0], [0, 1 / yscale, 0], [0, 0, 1]])
    )
    shear = affine_transform(np.array([[1, -1 / 2, 0], [0, 1, 0], [0, 0, 1]]))
    A, B = shear(scale(xy).T)
    C = 1.0 - (A + B)  # + (xscale-1) + (yscale-1)
    return np.vstack([A, B, C]).T
