import numpy as np
import pandas as pd
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def close(X: np.ndarray):
    if X.ndim == 2:
        return np.divide(X, np.sum(X, axis=1)[:, np.newaxis])
    else:
        return np.divide(X, np.sum(X, axis=0))


def renormalise(df: pd.DataFrame, components: list = [], scale=100.0):
    """
    Renormalises compositional data to ensure closure.

    Parameters:
    ------------
    df: pd.DataFrame
        Dataframe to renomalise.
    components: list
        Option subcompositon to renormalise to 100. Useful for the use case
        where compostional data and non-compositional data are stored in the
        same dataframe.
    scale: float, 100.
        Closure parameter. Typically either 100 or 1.
    """
    dfc = df.copy()
    if components:
        cmpnts = [c for c in components if c in dfc.columns]
        dfc.loc[:, cmpnts] = scale * dfc.loc[:, cmpnts].divide(
            dfc.loc[:, cmpnts].sum(axis=1).replace(0, np.nan), axis=0
        )
        return dfc
    else:
        dfc = dfc.divide(dfc.sum(axis=1).replace(0, 100.0), axis=0) * scale
        return dfc
