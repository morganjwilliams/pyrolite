"""
Functions to generate parameters for the construction of orthogonal polynomials which
are used to fit REE patterns.
"""
import numpy as np
import sympy.solvers.solvers
from sympy import symbols, var
from ..meta import update_docstring_references
from ...geochem.ind import REE, get_ionic_radii
from ..log import Handle

logger = Handle(__name__)


@update_docstring_references
def orthogonal_polynomial_constants(xs, degree=3, rounding=None, tol=10 ** -14):
    r"""
    Finds the parameters
    :math:`(\beta_0), (\gamma_0, \gamma_1), (\delta_0, \delta_1, \delta_2)` etc.
    for constructing orthogonal polynomial functions `f(x)` over a fixed set of values
    of independent variable `x`.
    Used for obtaining lambda values for dimensional reduction of REE data [#ref_1]_.

    Parameters
    ----------
    xs : :class:`numpy.ndarray`
        Indexes over which to generate the orthogonal polynomials.
    degree : :class:`int`
        Maximum polynomial degree. E.g. 2 will generate constant, linear, and quadratic
        polynomial components.
    tol : :class:`float`
        Convergence tolerance for solver.
    rounding : :class:`int`
        Precision for the orthogonal polynomial coefficents.

    Returns
    -------
    :class:`list`
        List of tuples corresponding to coefficients for each of the polynomial
        components. I.e the first tuple will be empty, the second will contain a single
        coefficient etc.

    Notes
    -----
        Parameters are used to construct orthogonal polymomials of the general form:

        .. math::

            f(x) &= a_0 \\
            &+ a_1 * (x - \beta) \\
            &+ a_2 * (x - \gamma_0) * (x - \gamma_1) \\
            &+ a_3 * (x - \delta_0) * (x - \delta_1) * (x - \delta_2) \\

    See Also
    --------
    :func:`~pyrolite.util.lambdas.calc_lambdas`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    References
    ----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__
    """
    xs = np.array(xs)
    x = var("x")
    params = []
    for d in range(degree):
        ps = symbols("{}0:{}".format(chr(945 + d), d))
        logger.debug("Generating {} DIM {} equations for {}.".format(d, d, ps))
        if d:
            eqs = []
            for _deg in range(d):
                q = 1
                if _deg:
                    q = x ** _deg
                for p in ps:
                    q *= x - p
                eqs.append(q)
            sums = []
            for q in eqs:
                sumq = 0.0
                for xi in xs:
                    sumq += q.subs(dict(x=xi))
                sums.append(sumq)

            guess = np.linspace(np.nanmin(xs), np.nanmax(xs), d + 2)[1:-1]
            result = sympy.solvers.solvers.nsolve(sums, ps, list(guess), tol=tol)
            if rounding is not None:
                result = np.around(np.array(result, dtype=np.float), decimals=rounding)
            params.append(tuple(result))
        else:
            params.append(())  # first parameter
    return params


def _get_tetrad_params(params=None):
    if params is None:
        params = ((58.75, 3.5), (62.25, 3.5), (65.75, 3.5), (69.25, 3.5))
    return params


def _get_params(params=None, degree=4):
    """
    Disambiguate parameter specification for orthogonal polynomials.

    Parameters
    ----------
    params : :class:`list` | :class:`str`
        Pre-computed parameters for the orthogonal polynomials (a list of tuples).
        Optionally specified, otherwise defaults the parameterisation as in
        O'Neill (2016). [#ref_1]_ If a string is supplied, :code:`"O'Neill (2016)"` or
        similar will give the original defaults, while :code:`"full"` will use all
        of the REE (including Eu) as a basis for the orthogonal polynomials.
    degree : :class:`int`
        Degree of orthogonal polynomial fit.

    Returns
    --------
    params : :class:`list`
        List of tuples containing a parameterisation of the orthogonal polynomial
        functions.
    """
    if params is None:
        # use standard parameters as used in O'Neill 2016 paper (exclude Eu)
        _ree = [i for i in REE() if i not in ["Eu"]]
        params = orthogonal_polynomial_constants(
            get_ionic_radii(_ree, charge=3, coordination=8),
            degree=degree,
        )
    elif isinstance(params, str):
        name = params.replace("'", "").lower()
        if "full" in name:
            # use ALL the REE for defininng the orthogonal polynomial functions
            # (include Eu)
            _ree = REE()
        elif ("oneill" in name) and ("2016" in name):
            # use standard parameters as used in O'Neill 2016 paper (exclude Eu)
            _ree = [i for i in REE() if i not in ["Eu"]]
        else:
            msg = "Parameter specification {} not recognised.".format(params)
            raise NotImplementedError(msg)
        params = orthogonal_polynomial_constants(
            get_ionic_radii(_ree, charge=3, coordination=8),
            degree=degree,
        )
    else:
        # check that params is a tuple or list
        if not isinstance(params, (list, tuple)):
            msg = "Type {} parameter specification {} not recognised.".format(
                type(params), params
            )
            raise NotImplementedError(msg)

    return params


def parse_sigmas(y, sigmas=None):
    r"""
    Disambigaute a value or set of sigmas for a dataset for use in lambda-fitting
    algorithms.

    Parameters
    ----------
    sigmas : :class:`float` | :class:`numpy.ndarray`
        2D array of REE uncertainties. Values as fractional uncertaintes
        (i.e. :math:`\sigma_{REE} / REE`).

    Returns
    -------
    sigmas : :class:`float` | :class:`numpy.ndarray`
        1D array of sigmas (:math:`\sigma_{REE} / REE`).

    Notes
    -----
    Note that the y-array is passed here only to infer the shape which should be
    assumed by the uncertainty array.
    Through propagation of uncertainties, the uncertainty on the natural logarithm of
    the normalised REE values are equivalent to :math:`\sigma_{REE} / REE` where the
    uncertainty in the reference composition is assumed to be zero. Thus, standard
    deviations of 1% in REE will result in :math:`\sigma=0.01` for the log-transformed
    REE. If no sigmas are provided, 1% uncertainty will be assumed and an array of
    0.01 will be returned.
    """
    sigma2d = False
    if sigmas is None:
        sigmas = np.ones(y.shape[1]) * 0.01
    else:  # sigmas are passed
        if isinstance(sigmas, float):
            sigmas = sigmas * np.ones(y.shape[1])
        elif sigmas.ndim > 1:
            if any(ix == 1 for ix in sigmas.shape):
                sigmas = sigmas.flatten()
            else:
                msg = "2D uncertainty estimation not yet implemented."
                raise NotImplementedError(msg)
        else:
            pass  # should be a 1D array

    return sigmas
