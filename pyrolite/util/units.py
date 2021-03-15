import numpy as np
import pandas as pd
from .log import Handle

logger = Handle(__name__)


__massunits__ = {
    "%": 10 ** -2,
    "pct": 10 ** -2,
    "wt%": 10 ** -2,
    "ppm": 10 ** -6,
    "ppb": 10 ** -9,
    "ppt": 10 ** -12,
    "ppq": 10 ** -15,
}

__UNITS__ = {**__massunits__}


def scale(in_unit, target_unit="ppm"):
    """
    Provides the scale difference between two mass units.

    Parameters
    ----------
    in_unit : :class:`str`
        Units to be converted from
    target_unit : :class:`str`, :code:`"ppm"`
        Units to scale to.

    Todo
    -------
        * Implement different inputs: :class:`str`, :class:`list`, :class:`pandas.Series`

    Returns
    --------
    :class:`float`
    """
    in_unit = str(in_unit).lower()
    target_unit = str(target_unit).lower()
    if (
        not pd.isna(in_unit)
        and (in_unit in __UNITS__.keys())
        and (target_unit in __UNITS__.keys())
    ):
        scale = __UNITS__[in_unit] / __UNITS__[target_unit]
    else:
        unkn = [i for i in [in_unit, target_unit] if i not in __UNITS__]
        logger.warning("Units not known: {}. Defaulting to unity.".format(unkn))
        scale = 1.0
    return scale
