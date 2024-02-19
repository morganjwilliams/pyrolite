import pandas as pd


def _collect_lambda_outputs(
    B, s, X2, src, names, add_uncertainties=False, add_X2=False
):
    """
    Collect estimates, uncertainties and Chi-squared values for lambda
    and tetrad parameter estimates.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Parameter estimates.
    s : :class:`numpy.ndarray`
        Parameter uncertainty estimates.
    X2 : :class:`numpy.ndarray`
        Chi-squared values.
    src : :class:`pandas.DataFrame` | :class:`pandas.Series
        The original source data.
    names : :class:`list`
        Names for the parameters.
    add_uncertainties : :class:`bool`
        Whether to append Chi-squared values.
    add_X2 : :class:`bool`
        Whether to append Chi-squared values.
    """
    if src.ndim > 1:
        lambdas = pd.DataFrame(B, index=src.index, columns=names, dtype=B.dtype)
    else:
        if B.ndim > 1:
            B, s, X2 = B[0], s[0], X2[0]
        lambdas = pd.Series(
            B,
            index=names,
            name=src.name,
            dtype=B.dtype,
        )
    if add_uncertainties:
        lambdas[[n + "_" + chr(963) for n in names]] = s
    if add_X2:
        lambdas["X2"] = X2
    return lambdas
