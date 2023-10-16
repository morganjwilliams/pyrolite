"""
Linear algebra methods for fitting a series of orthogonal polynomial functions to
REE patterns.
"""
import numpy as np
import pandas as pd

from ..log import Handle
from ..meta import update_docstring_references
from ..missing import md_pattern
from .eval import get_function_components, lambda_poly
from .params import parse_sigmas
from .helpers import b_s_x2_to_df, b_s_x2_to_series

logger = Handle(__name__)


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
    :func:`~pyrolite.util.lambdas.params.orthogonal_polynomial_constants`
    """
    radii = np.array(radii)
    degree = len(params)
    a = np.vander(radii, degree, increasing=True).T
    b = np.array([lambda_poly(radii, pset) for pset in params])
    A_radii = a[:, np.newaxis, :] * b[np.newaxis, :, :]
    A = A_radii.sum(axis=-1)  # `A` as in O'Neill (2016)
    return A


def lambdas_ONeill2016_series(
    series,
    radii,
    params=None,
    sigmas=None,
    add_X2=False,
    add_uncertainties=False,
    **kwargs
):
    r"""
    This is a variant of :func:`~pyrolite.geochem.lambdas_ONeill2016`
    adapted to work on a Series instead of a DataFrame.
    Implementation of the original algorithm. [#ref_1]_

    Parameters
    -----------
    series : :class:`pandas.Series`
        Series of REE data, with sample analyses organised by row.
    radii : :class:`list`, :class:`numpy.ndarray`
        Radii at which to evaluate the orthogonal polynomial.
    params : :class:`tuple`
        Tuple of constants for the orthogonal polynomial.
    sigmas : :class:`float` | :class:`numpy.ndarray`
        Single value or 1D array of normalised observed value uncertainties
        (:math:`\sigma_{REE} / REE`).
    add_uncertainties : :class:`bool`
        Append parameter standard errors to the dataframe.

    Returns
    --------
    :class:`pandas.Series`

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.params.orthogonal_polynomial_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    References
    -----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__

    """
    assert params is not None
    names, x0, func_components = get_function_components(radii, params=params)
    X = np.array(func_components).T
    sigmas = parse_sigmas(series.size, sigmas=sigmas)

    xd = len(func_components)
    rad = np.array(radii)

    B = np.ones(xd) * np.nan
    s = np.ones(xd) * np.nan

    missing_fltr = np.isfinite(series)
    numberOfPresentElemetns = missing_fltr.sum()

    A = get_polynomial_matrix(rad[missing_fltr], params=params)
    invA = np.linalg.inv(A)
    V = np.vander(rad, xd, increasing=True).T
    Z = (series.values[missing_fltr][np.newaxis, :] * V[:, missing_fltr]).sum(axis=-1)
    _B = (invA @ Z.T).T
    ############################################################################
    _sigmas = sigmas[missing_fltr]
    _x = X[missing_fltr, :]
    W = np.eye(_sigmas.shape[0]) * 1 / _sigmas**2  # weights
    invXWX = np.linalg.inv(_x.T @ W @ _x)

    est = (X[missing_fltr] @ _B.T).T  # modelled values
    # residuals over all rows
    residuals = (series.loc[missing_fltr] - est).values
    dof = (
        numberOfPresentElemetns - xd
    )  # effective degrees of freedom (for this mising filter)
    # chi-sqared as SSQ / sigmas / residual degrees of freedom
    reduced_chi_squared = (residuals**2 / _sigmas**2).sum() / dof
    _s = np.sqrt(reduced_chi_squared.reshape(-1, 1) * np.diag(invXWX))

    B[:] = _B
    s[:] = _s
    return b_s_x2_to_series(B, s, reduced_chi_squared, names, add_uncertainties, add_X2)


@update_docstring_references
def lambdas_ONeill2016(
    df, radii, params=None, sigmas=None, add_X2=False, add_uncertainties=False, **kwargs
):
    r"""
    Implementation of the original algorithm. [#ref_1]_

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe of REE data, with sample analyses organised by row.
    radii : :class:`list`, :class:`numpy.ndarray`
        Radii at which to evaluate the orthogonal polynomial.
    params : :class:`tuple`
        Tuple of constants for the orthogonal polynomial.
    sigmas : :class:`float` | :class:`numpy.ndarray`
        Single value or 1D array of normalised observed value uncertainties
        (:math:`\sigma_{REE} / REE`).
    add_uncertainties : :class:`bool`
        Append parameter standard errors to the dataframe.

    Returns
    --------
    :class:`pandas.DataFrame`

    See Also
    ---------
    :func:`~pyrolite.util.lambdas.params.orthogonal_polynomial_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    References
    -----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__

    """
    assert params is not None
    names, x0, func_components = get_function_components(radii, params=params)
    X = np.array(func_components).T
    y = df.values
    sigmas = parse_sigmas(y.shape[1], sigmas=sigmas)

    xd = len(func_components)
    rad = np.array(radii)  # so we can use a boolean index

    B = np.ones((y.shape[0], xd)) * np.nan
    s = np.ones((y.shape[0], xd)) * np.nan
    χ2 = np.ones((y.shape[0], 1)) * np.nan
    md_inds, patterns = md_pattern(df)
    # for each missing data pattern, we create the matrix A - rather than each row
    for ind in np.unique(md_inds):
        row_fltr = md_inds == ind  # rows with this pattern
        missing_fltr = ~patterns[ind]["pattern"]  # boolean presence-absence filter
        if missing_fltr.sum():  # ignore completely empty rows
            yd = missing_fltr.sum()  # number of elements used for the fit
            A = get_polynomial_matrix(rad[missing_fltr], params=params)
            invA = np.linalg.inv(A)
            V = np.vander(rad, xd, increasing=True).T
            Z = (
                y[np.ix_(row_fltr, missing_fltr)][:, np.newaxis]
                * V[np.newaxis, :, missing_fltr]
            ).sum(axis=-1)
            _B = (invA @ Z.T).T

            ############################################################################
            _sigmas = sigmas[missing_fltr]
            _x = X[missing_fltr, :]
            W = np.eye(_sigmas.shape[0]) * 1 / _sigmas**2  # weights
            invXWX = np.linalg.inv(_x.T @ W @ _x)

            est = (X[missing_fltr, :] @ _B.T).T  # modelled values
            # residuals over all rows
            residuals = (df.loc[row_fltr, missing_fltr] - est).values
            dof = yd - xd  # effective degrees of freedom (for this mising filter)
            # chi-sqared as SSQ / sigmas / residual degrees of freedom
            reduced_chi_squared = (residuals**2 / _sigmas**2).sum(axis=1) / dof
            _s = np.sqrt(reduced_chi_squared.reshape(-1, 1) * np.diag(invXWX))

            B[row_fltr, :] = _B
            s[row_fltr, :] = _s
            χ2[row_fltr, 0] = reduced_chi_squared
    return b_s_x2_to_df(B, s, χ2, df.index, names, add_uncertainties, add_X2)
