"""
Transformation utilites for matplotlib.
"""
import numpy as np
from ...comp.codata import close
from ..log import Handle

logger = Handle(__name__)


def affine_transform(mtx=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    """
    Construct a function which will perform a 2D affine transform based on
    a 3x3 affine matrix.

    Parameters
    -----------
    mtx : :class:`numpy.ndarray`
    """

    def tfm(data):
        xy = data[:, :2]
        return (mtx @ np.vstack((xy.T[:2], np.ones(xy.T.shape[1]))))[:2]

    return tfm


def ABC_to_xy(ABC, xscale=1.0, yscale=1.0):
    """
    Convert ternary compositional coordiantes to x-y coordinates
    for visualisation within a triangle.

    Parameters
    -----------
    ABC : :class:`numpy.ndarray`
        Ternary array (:code:`samples, 3`).
    xscale : :class:`float`
        Scale for x-axis.
    yscale : :class:`float`
        Scale for y-axis.

    Returns
    --------
    :class:`numpy.ndarray`
        Array of x-y coordinates (:code:`samples, 2`)
    """
    assert ABC.shape[-1] == 3
    # transform from ternary to xy cartesian
    scale = affine_transform(np.array([[xscale, 0, 0], [0, yscale, 0], [0, 0, 1]]))
    shear = affine_transform(np.array([[1, 1 / 2, 0], [0, 1, 0], [0, 0, 1]]))
    xy = scale(shear(close(ABC)).T)
    return xy.T


def xy_to_ABC(xy, xscale=1.0, yscale=1.0):
    """
    Convert x-y coordinates within a triangle to compositional ternary coordinates.

    Parameters
    -----------
    xy : :class:`numpy.ndarray`
        XY array (:code:`samples, 2`).
    xscale : :class:`float`
        Scale for x-axis.
    yscale : :class:`float`
        Scale for y-axis.

    Returns
    --------
    :class:`numpy.ndarray`
        Array of ternary coordinates (:code:`samples, 3`)
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
