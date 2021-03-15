"""
Generation and evalutation of orthogonal polynomial and tetrad functions from sets of
parameters (the sequence of polymomial roots and tetrad centres and widths).
"""
import numpy as np

from .transform import REE_radii_to_z
from .params import orthogonal_polynomial_constants, _get_params, _get_tetrad_params
from ..log import Handle

logger = Handle(__name__)



def lambda_poly(x, ps):

    """
    Evaluate polynomial `lambda_n(x)` given a tuple of parameters `ps` with length
    equal to the polynomial degree.

    Parameters
    -----------
    x : :class:`numpy.ndarray`
        X values to calculate the function at.
    ps: :class:`tuple`
        Parameter set tuple. E.g. parameters `(a, b)` from :math:`f(x) = (x-a)(x-b)`.

    Returns
    --------
    :class:`numpy.ndarray`
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    result = np.ones(len(x))
    for p in ps:
        result = result * (x - p)
    return result.astype(np.float)


def tetrad(x, centre, width):
    """
    Evaluate :math:`f(z)` describing a tetrad with specified centre and width.

    Parameters
    ----------
    x
    centre : :class:`float`

    width : :class:`float`

    Returns
    --------
    """
    g = (x - centre) / (width / 2)
    x0 = 1 - g ** 2
    tet = (x0 + np.sqrt(x0 ** 2)) / 2
    tet[tet < 0] = 0
    return tet


def get_tetrads_function(params=None):
    params = _get_tetrad_params(params=params)

    def tetrads(x, sum=True):
        ts = np.array([tetrad(x, centre, width) for centre, width in params])
        if sum:
            ts = np.sum(ts, axis=0)
        return ts

    return tetrads


def get_lambda_poly_function(lambdas: np.ndarray, params=None, radii=None, degree=5):
    """
    Expansion of lambda parameters back to the original space. Returns a
    function which evaluates the sum of the orthogonal polynomials at given
    `x` values.

    Parameters
    ------------
    lambdas: :class:`numpy.ndarray`
        Lambda values to weight combination of polynomials.
    params: :class:`list`(:class:`tuple`)
        Parameters for the orthogonal polynomial decomposition.
    radii: :class:`numpy.ndarray`
        Radii values used to construct the lambda values. [#note_1]_
    degree: :class:`int`
        Degree of the orthogonal polynomial decomposition. [#note_1]_

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.calc_lambdas`
    :func:`~pyrolite.util.lambdas.params.orthogonal_polynomial_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    Notes
    -----
        .. [#note_1] Only needed if parameters are not supplied
    """
    if params is None and radii is not None:
        params = orthogonal_polynomial_constants(radii, degree=degree)
    elif params is None and radii is None:
        msg = """Must provide either x values to construct parameters,
                 or the parameters themselves."""
        raise AssertionError(msg)

    def _lambda_evaluator(xarr):
        """
        Calculates the sum of decomposed polynomial components
        at given x values.

        Parameters
        -----------
        xarr: :class:`numpy.ndarray`
            X values at which to evaluate the function.
        """
        func_components = np.array([lambda_poly(xarr, pset) for pset in params])
        return np.dot(lambdas, func_components)

    return _lambda_evaluator


def get_function_components(
    radii, params=None, fit_tetrads=False, tetrad_params=None, degree=5, **kwargs
):
    lambda_params = _get_params(params=params, degree=degree)
    degree = len(lambda_params)
    names = [chr(955) + str(d) for d in range(degree)]
    func_components = [lambda_poly(radii, pset) for pset in lambda_params]
    x0 = list(np.exp(np.arange(degree) + 2) / 2)
    if fit_tetrads:
        zs = REE_radii_to_z(radii)
        if tetrad_params is None:
            tetrad_params = [(c, 3.5) for c in [58.75, 62.25, 65.75, 69.25]]
        func_components += list(
            get_tetrads_function(params=tetrad_params)(zs, sum=False)
        )

        names += [chr(964) + str(d) for d in range(4)]

        x0 += [0, 0, 0, 0]  # expect tetrads to be small to zero
    return names, x0, func_components
