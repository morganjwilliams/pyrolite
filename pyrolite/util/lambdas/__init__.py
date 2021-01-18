import numpy as np
import pandas as pd
from sympy.solvers.solvers import nsolve
from sympy import symbols, var
import scipy.optimize
import scipy.linalg
from ...geochem.ind import REE, get_ionic_radii
from ... import plot
from ..meta import update_docstring_references
from ..missing import md_pattern
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


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
    :func:`~pyrolite.util.math.lambdas`
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
            result = nsolve(sums, ps, list(guess), tol=tol)
            if rounding is not None:
                result = np.around(np.array(result, dtype=np.float), decimals=rounding)
            params.append(tuple(result))
        else:
            params.append(())  # first parameter
    return params


########################################################################################


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


########################################################################################


def get_polynomial_matrix(radii, params=None):
    """
    Create the matrix `A` with polynomial components across the columns,
    and increasing order down the rows.

    Parameters
    -----------
    radii : :class:`list`, :class:`numpy.ndarray`
        Radii at which to evaluate the orthogonal polynomial.
    params : :class:`tuple`
        Tuple of constants for the orthogonal polynomial.

    Returns
    --------
    :class:`numpy.ndarray`

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.orthogonal_polynomial_constants`
    """
    radii = np.array(radii)
    degree = len(params)
    a = np.vander(radii, degree, increasing=True).T
    b = np.array([evaluate_lambda_poly(radii, pset) for pset in params])
    A_radii = a[:, np.newaxis, :] * b[np.newaxis, :, :]
    A = A_radii.sum(axis=-1)  # `A` as in O'Neill (2016)
    return A


@update_docstring_references
def lambdas_ONeill2016(df, radii, params=None):
    """
    Implementation of the original algorithm. [#ref_1]_

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe of REE data, with sample analyses organised by row.
    radii : :class:`list`, :class:`numpy.ndarray`
        Radii at which to evaluate the orthogonal polynomial.
    params : :class:`tuple`
        Tuple of constants for the orthogonal polynomial.

    Returns
    --------
    :class:`pandas.DataFrame`

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.orthogonal_polynomial_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    References
    -----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__

    Todo
    -----
    Use a missing-data-pattern approach to speed up the algorithm slightly (i.e.
    generate an :code:`A` matrix for each pattern).
    """
    assert params is not None
    degree = len(params)
    # initialise the dataframe
    lambdas = pd.DataFrame(
        index=df.index,
        columns=[chr(955) + str(d) for d in range(degree)],
        dtype="float32",
    )
    rad = np.array(radii)  # so we can use a boolean index

    md_inds, patterns = md_pattern(df)
    # for each missing data pattern, we create the matrix A - rather than each row
    for ind in np.unique(md_inds):
        row_fltr = md_inds == ind  # rows with this pattern
        missing_fltr = ~patterns[ind]["pattern"]  # boolean presence-absence filter
        if missing_fltr.sum():  # ignore completely empty rows
            A = get_polynomial_matrix(rad[missing_fltr], params=params)
            invA = np.linalg.inv(A)
            V = np.vander(rad, degree, increasing=True).T
            Z = (
                df.loc[row_fltr, missing_fltr].values[:, np.newaxis]
                * V[np.newaxis, :, missing_fltr]
            ).sum(axis=-1)
            lambdas.loc[row_fltr, :] = (invA @ Z.T).T
    return lambdas


########################################################################################


def _cost_func(ls, ys, func_components, power=1.0):
    """
    Cost function for lambda optimization.

    Parameters
    ------------
    ls : :class:`numpy.ndarray`
        Lambda values, effectively weights for the polynomial components.
    ys : :class:`numpy.ndarray`
        Target y values.
    func_components : :class:`numpy.ndarray`
        Arrays representing the individual unweighted function components.
        E.g. :code:`[[a, a, ...], [x - b, x - b, ...], ...]` for lambdas.
    power : :class:`float`
        Power for the cost function.

    Returns
    -------
    :class:`numpy.ndarray`
        Cost at the given set of `ls`.
    """
    cost = np.abs(ls @ func_components - ys) ** power
    cost[np.isnan(cost)] = 0.0  # can't change nans - don't penalise them
    return cost


def _residuals_func(ls, ys, func_components):
    """
    Residuals function for lambda optimization.

    Parameters
    ------------
    ls : :class:`numpy.ndarray`
        Lambda values, effectively weights for the polynomial components.
    ys : :class:`numpy.ndarray`
        Target y values.
    func_components : :class:`numpy.ndarray`
        Arrays representing the individual unweighted unweighted function components.
        E.g. :code:`[[a, a, ...], [x - b, x - b, ...], ...]` for lambdas.

    Returns
    -------
    :class:`numpy.ndarray`
        Residuals at the given set of `ls`.
    """
    res = ls @ func_components - ys
    res[np.isnan(res)] = 0.0  # can't change nans - don't penalise them
    return res


@update_docstring_references
def lambdas_optimize(
    df: pd.DataFrame,
    radii,
    params=None,
    guess=None,
    degree=5,
    residuals_function=_residuals_func,
):
    """
    Parameterises values based on linear combination of orthogonal polynomials
    over a given set of values for independent variable `x`. [#ref_1]_

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Target data to fit. For geochemical data, this is typically normalised
        so we can fit a smooth function.
    radii : :class:`list`, :class:`numpy.ndarray`
        Radii at which to evaluate the orthogonal polynomial.
    params : :class:`list`, :code:`None`
        Orthogonal polynomial coefficients (see
        :func:`orthogonal_polynomial_constants`).
    residuals_function : :class:`Callable`
        Residuals function to use for optimization of lambdas.

    Returns
    --------
    :class:`numpy.ndarray` | (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        Optimial results for weights of orthogonal polymomial regression (`lambdas`).

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.orthogonal_polynomial_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    References
    ----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__
    """
    assert params is not None
    degree = len(params)
    lambdas = pd.DataFrame(
        index=df.index,
        columns=[chr(955) + str(d) for d in range(degree)],
        dtype="float32",
    )
    starting_guess = np.exp(np.arange(degree) + 2) / 2
    # arrays representing the unweighted individual polynomial components
    func_components = np.array([evaluate_lambda_poly(radii, pset) for pset in params])

    for row in range(df.index.size):
        result = scipy.optimize.least_squares(
            residuals_function,
            starting_guess,
            args=(
                df.iloc[row, :].values,
                func_components,
            ),
        )
        x = result.x
        # redisuals res = result.fun
        lambdas.iloc[row, :] = x
    return lambdas


########################################################################################
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


@update_docstring_references
def calc_lambdas(df, params=None, degree=4, exclude=[], algorithm="ONeill", **kwargs):
    """
    Parameterises values based on linear combination of orthogonal polynomials
    over a given set of values for independent variable `x` [#ref_1]_ .
    This function expects to recieve data already normalised and log-transformed.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        Dataframe containing REE Data.
    params : :class:`list` | :class:`str`
        Pre-computed parameters for the orthogonal polynomials (a list of tuples).
        Optionally specified, otherwise defaults the parameterisation as in
        O'Neill (2016). [#ref_1]_ If a string is supplied, :code:`"O'Neill (2016)"` or
        similar will give the original defaults, while :code:`"full"` will use all
        of the REE (including Eu) as a basis for the orthogonal polynomials.
    degree : :class:`int`
        Degree of orthogonal polynomial fit.
    exclude : :class:`list`
        REE to exclude from the *fit*.
    algorithm : :class:`str`
        Algorithm to use for fitting the orthogonal polynomials.

    Returns
    --------
    :class:`pd.DataFrame`

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.orthogonal_polynomial_constants`
    :func:`~pyrolite.geochem.pyrochem.lambda_lnREE`
    :func:`~pyrolite.geochem.pyrochem.normalize_to`

    References
    ----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__
    """

    # parameters should be set here, and only once; these define the inividual
    # orthogonal polynomial functions which are combined to compose the REE pattern
    params = _get_params(params=params, degree=degree)
    # these are the REE which the lambdas will be EVALUATED at; exclude empty columns
    columns = [c for c in df.columns if c not in exclude and np.isfinite(df[c]).sum()]
    if not columns:
        msg = "No columns specified (after exclusion), nothing to calculate."
        raise IndexError(msg)
    radii = get_ionic_radii(columns, charge=3, coordination=8)

    df = df[columns]
    df[~np.isfinite(df)] = np.nan  # deal with np.nan, np.inf

    if "oneill" in algorithm.lower():
        try:
            return lambdas_ONeill2016(df, radii=radii, params=params, **kwargs)
        except np.linalg.LinAlgError:  # singular matrix, use optimize
            return lambdas_optimize(df, radii=radii, params=params, **kwargs)
    else:
        return lambdas_optimize(df, radii=radii, params=params, **kwargs)


def pcov_from_jac(jac):
    """
    Extract a covariance matrix from a Jacobian matrix returned from
    :mod:`scipy.optimize` functions.

    Parameters
    ----------
    jac : :class:`numpy.ndarray`
        Jacobian array.

    Returns
    -------
    pcov : :class:`numpy.ndarray`
        Square covariance array; this hasn't yet been scaled by residuals.
    """
    # from scipy.opt minpack
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = scipy.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    return pcov


def fit_components(y, x0, func_components, residuals_function=_residuals_func):
    """
    Fit a weighted sum of function components using
    :func:`scipy.optimize.least_squares`.

    Parameters
    -----------
    y : :class:`numpy.ndarray`
        Array of target values to fit.
    x0 : :class:`numpy.ndarray`
        Starting guess for the function weights.
    func_components : :class:`list` ( :class:`numpy.ndarray` )
        List of arrays representing static/evaluated function components.
    redsiduals_function : callable
        Callable funciton to compute residuals which accepts ordered arguments for
        weights, target values and function components.

    Returns
    -------
    arr, uarr : :class:`numpy.ndarray`
        Arrays for the optimized parameter values (arr) and parameter
        uncertaintes (uarr, 1σ).
    """
    m, n = y.shape[0], x0.size  # shape of output
    arr = np.ones((m, n)) * np.nan
    uarr = np.ones((m, n)) * np.nan
    for row in range(m):
        res = scipy.optimize.least_squares(
            residuals_function,
            x0,
            args=(
                y[row, :],
                func_components,
            ),
        )
        arr[row, :] = res.x
        # get the covariance matrix of the parameters from the jacobian
        pcov = pcov_from_jac(res.jac)
        yd, xd = y.shape[0], x0.size
        if yd > xd:
            s_sq = res.cost / (yd - xd)  # check rank
            pcov = pcov * s_sq
        else:
            pcov.fill(inf)
        uarr[row, :] = np.sqrt(np.diag(pcov))  # sigmas on parameters
    return arr, uarr


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


def REE_z_to_radii(z, fit=None, degree=7):
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
        radii = np.array(get_ionic_radii(REE(dropPm=False), charge=3, coordination=8))
        p, resids, rank, s, rcond = np.polyfit(
            np.arange(57, 72), radii, degree, full=True
        )

        def fit(x):
            return np.polyval(p, x)

    r = fit(z)
    return r


def REE_radii_to_z(r, fit=None, degree=7):
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
        Approximate atomic nubmers for given radii.
    """
    if fit is None:
        radii = np.array(get_ionic_radii(REE(dropPm=False), charge=3, coordination=8))
        p, resids, rank, s, rcond = np.polyfit(
            radii, np.arange(57, 72), degree, full=True
        )

        def fit(x):
            return np.polyval(p, x)

    z = fit(r)
    return z


def plot_lambdas_components(lambdas, ax=None, params=None, degree=4, **kwargs):
    """
    Plot a decomposed orthogonal polynomial using the lambda coefficients.

    Parameters
    ----------
    lambdas
        1D array of lambdas.
    ax : :class:`matplotlib.axes.Axes`
        Axis to plot on.

    Returns
    --------
    :class:`matplotlib.axes.Axes`
    """
    radii = np.array(get_ionic_radii(REE(), charge=3, coordination=8))
    xs = np.linspace(np.max(radii), np.min(radii), 100)
    if params is None:
        if params is None:  # use standard parameters as used in O'Neill 2016 paper
            default_REE = [i for i in REE() if i not in ["Eu"]]
            default_radii = get_ionic_radii(default_REE, charge=3, coordination=8)
            params = orthogonal_polynomial_constants(default_radii, degree=degree)
    else:
        params = params[: degree - 1]  # limit the degree for the vis
    ax = plot.spider.REE_v_radii(ax=ax)
    # plot the overall function
    overall_func = get_lambda_poly_func(lambdas, params)
    ax.plot(  # plot the polynomials
        xs, overall_func(xs), label="Regression", color="k", **kwargs
    )
    for w, p in zip(lambdas, params):  # plot the components
        l_func = get_lambda_poly_func([w], [p])  # pasing singluar vaules and one tuple
        label = (
            "$r^{}: \lambda_{}".format(len(p), len(p))
            + ["\cdot f_{}".format(len(p)), ""][int(len(p) == 0)]
            + "$"
        )
        ax.plot(xs, l_func(xs), label=label, ls="--", **kwargs)  # plot the polynomials
    return ax
