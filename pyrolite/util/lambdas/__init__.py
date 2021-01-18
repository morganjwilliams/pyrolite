import numpy as np
import pandas as pd
from ...geochem.ind import REE, get_ionic_radii
from ..meta import update_docstring_references
from ..log import Handle

from .params import orthogonal_polynomial_constants, _get_params
from .oneill import lambdas_ONeill2016
from .opt import lambdas_optimize
from .tetrads import get_tetrads_function, tetrad
from .plot import plot_lambdas_components, plot_tetrads_profiles
from .transform import REE_radii_to_z, REE_z_to_radii

logger = Handle(__name__)


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
