import numpy as np
import pandas as pd


def iscollection(obj) -> bool:
    """
    Checks whether an object is an iterable collection.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    -------
    bool
        Boolean indication of whether the object is a collection.
    """
    return isinstance(obj, (list, np.ndarray, set, tuple, dict, pd.Series))
