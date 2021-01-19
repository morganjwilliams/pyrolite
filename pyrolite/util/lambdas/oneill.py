"""
Linear algebra methods for fitting a series of orthogonal polynomial functions to
REE patterns.
"""
import numpy as np
import pandas as pd
from .eval import lambda_poly
from ..missing import md_pattern
from ..meta import update_docstring_references
from ..log import Handle

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
    :func:`~pyrolite.util.lambdas.params.orthogonal_polynomial_constants`
    :func:`~pyrolite.geochem.transform.lambda_lnREE`

    References
    -----------
    .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
           Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
           doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__

    """
    assert params is not None
    degree = len(params)
    # initialise the dataframe
    names = [chr(955) + str(d) for d in range(degree)]
    lambdas = pd.DataFrame(index=df.index, columns=names, dtype="float32")
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
