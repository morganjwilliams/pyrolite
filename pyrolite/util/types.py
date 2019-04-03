import numpy as np
import pandas as pd


def iscollection(obj):
    """
    Checks whether an object is an iterable collection.

    Parameters
    ----------
    obj : :class:`object`
        Object to check.

    Returns
    -------
    :class:`bool`
        Boolean indication of whether the object is a collection.
    """

    for ty in [list, np.ndarray, set, tuple, dict, pd.Series]:
        if isinstance(obj, ty):
            return True

    return False
