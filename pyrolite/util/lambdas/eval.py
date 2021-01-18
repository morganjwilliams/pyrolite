"""
Generation and evalutation of orthogonal polynomial functions from a set of parameters
(the sequence of polymomial roots).
"""
import numpy as np
from ..log import Handle
from .params import orthogonal_polynomial_constants

logger = Handle(__name__)


def evaluate_lambda_poly(x, ps):

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


def get_lambda_poly_func(lambdas: np.ndarray, params=None, radii=None, degree=5):
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
    :func:`~pyrolite.util.lambdas.lambdas`
    :func:`~pyrolite.util.lambdas.orthogonal_polynomial_constants`
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
        func_components = np.array(
            [evaluate_lambda_poly(xarr, pset) for pset in params]
        )
        return np.dot(lambdas, func_components)

    return _lambda_evaluator
