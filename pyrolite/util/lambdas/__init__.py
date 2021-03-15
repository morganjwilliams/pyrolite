import numpy as np
import pandas as pd
from ...geochem.ind import REE, get_ionic_radii
from ..meta import update_docstring_references

from .params import orthogonal_polynomial_constants, _get_params
from .oneill import lambdas_ONeill2016
from .opt import lambdas_optimize
from .plot import plot_lambdas_components, plot_profiles
from .transform import REE_radii_to_z, REE_z_to_radii
from .eval import get_function_components
from ..log import Handle

logger = Handle(__name__)


@update_docstring_references
def calc_lambdas(
    df,
    params=None,
    degree=4,
    exclude=[],
    algorithm="ONeill",
    anomalies=[],
    fit_tetrads=False,
    sigmas=None,
    add_uncertainties=False,
    add_X2=False,
    **kwargs
):
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
    anomalies : :class:`list`
        List of relative anomalies to append to the dataframe.
    fit_tetrads : :class:`bool`
        Whether to fit tetrad functions in addition to orthogonal polynomial functions.
        This will force the use of the optimization algorithm.
    sigmas : :class:`float` | :class:`numpy.ndarray`
        Single value or 1D array of observed value uncertainties.
    add_uncertainties : :class:`bool`
        Whether to append estimated parameter uncertainties to the dataframe.
    add_X2 : :class:`bool`
        Whether to append the chi-squared values (χ2) to the dataframe.

    Returns
    --------
    :class:`pd.DataFrame`

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.params.orthogonal_polynomial_constants`
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
    if fit_tetrads and ("oneill" in algorithm.lower()):
        logger.warning(
            "Can't use the O'Neill (2016) algorithm to fit tetrads; "
            "falling back to the optimization based algorithm."
        )
        algorithm = "opt"
    # this is what will be passed to the fit
    #  to cacluate an anomaly rather than a residual, exclude the element from the fit
    exclude += anomalies
    if exclude:
        logger.debug("Excluding columns from the fit: " + ",".join(exclude))
    # these are the REE which the lambdas will be EVALUATED at; exclude empty columns
    column_fltr = [
        (c not in exclude) and (np.isfinite(df[c]).sum() > 0) for c in df.columns
    ]
    columns = df.columns[column_fltr].tolist()
    if not columns:
        msg = "No columns specified (after exclusion), nothing to calculate."
        raise IndexError(msg)

    # also filter the sigmas we pass to subsequent functions, if needed
    if not (sigmas is None):
        logger.debug("Sigmas provided.")
        if not isinstance(sigmas, (int, float)):
            sigmas = sigmas[column_fltr]

    fit_df = df.loc[:, columns]
    fit_df.mask(~np.isfinite(fit_df), np.nan, inplace=True)  # deal with np.nan, np.inf
    fit_radii = get_ionic_radii(columns, charge=3, coordination=8)

    if "oneill" in algorithm.lower():
        logger.debug("Using implementation of ONeill2016.")
        ls = lambdas_ONeill2016(
            fit_df,
            radii=fit_radii,
            params=params,
            add_uncertainties=add_uncertainties,
            add_X2=add_X2,
            sigmas=sigmas,
            **kwargs
        )
    else:
        logger.debug("Using optimization algorithm.")
        ls = lambdas_optimize(
            fit_df,
            radii=fit_radii,
            params=params,
            fit_tetrads=fit_tetrads,
            add_uncertainties=add_uncertainties,
            add_X2=add_X2,
            sigmas=sigmas,
            **kwargs
        )
    if anomalies:
        logger.debug("Calculating anomalies.")
        # radii here use all the REE columns in df, including those excluded
        ree = df.pyrochem.list_REE
        names, x0, func_components = get_function_components(
            get_ionic_radii(ree, charge=3, coordination=8),
            params=params,
            fit_tetrads=fit_tetrads,
            **kwargs
        )
        npars = ls.columns.size - int(add_X2)  # exclude chi_squared if included
        npars -= int(add_uncertainties) * npars // 2  # remove parameter uncertainties

        regression = pd.DataFrame(
            ls.loc[:, ls.columns[:npars]].values @ np.array(func_components),
            columns=ree,
            index=df.index,
        )

        rdiff = df[ree] - regression

        for anomaly in anomalies:  # add anomalies in linear (not log) space
            assert anomaly in rdiff.columns
            # log residuals are linear ratios, can back-transform
            ls["{}/{}*".format(anomaly, anomaly)] = np.exp(rdiff[anomaly])
    return ls
