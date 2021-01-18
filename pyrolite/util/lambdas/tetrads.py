import numpy as np
from ..log import Handle

logger = Handle(__file__)


def tetrad(centre, width):
    """
    Generate a function :math:`f(z)` describing a tetrad given a specified centre and
    width.

    Parameters
    ----------
    centre : :class:`float`

    width : :class:`float`

    Returns
    --------
    """

    def tet(x):
        g = (x - centre) / (width / 2)
        x0 = 1 - g ** 2
        r = (x0 + np.sqrt(x0 ** 2)) / 2
        r[r < 0] = 0
        return r

    return tet


def get_tetrads_function(params=None):
    if params is None:
        params = ((58.75, 3.5), (62.25, 3.5), (65.75, 3.5), (69.25, 3.5))

    def tetrads(x, sum=True):
        ts = np.array([tetrad(centre, width)(x) for centre, width in params])
        if sum:
            ts = np.sum(ts, axis=0)
        return ts

    return tetrads
