import pandas as pd

from .log import Handle

logger = Handle(__name__)


__massunits__: dict[str, float] = {
    "%": 10**-2,
    "pct": 10**-2,
    "wt%": 10**-2,
    "ppm": 10**-6,
    "ppb": 10**-9,
    "ppt": 10**-12,
    "ppq": 10**-15,
}

__UNITS__: dict[str, float] = {**__massunits__}


def scale(in_unit, target_unit="ppm"):
    """
    Provides the scale difference between two mass units.

    Parameters
    ----------
    in_unit : str
        Units to be converted from
    target_unit : str
        Units to scale to.

    Todo
    -------
        * Implement different inputs: `str`, `list`, `pandas.Series`

    Returns
    --------
    float
    """
    in_unit: str = str(in_unit).lower()
    target_unit: str = str(target_unit).lower()
    if not pd.isna(in_unit) and (in_unit in __UNITS__) and (target_unit in __UNITS__):
        scale: float = __UNITS__[in_unit] / __UNITS__[target_unit]
    else:
        unkn: list[str] = [i for i in [in_unit, target_unit] if i not in __UNITS__]
        logger.info(f"Units not known: {unkn}. Defaulting to unity.")
        scale = 1.0
    return scale
