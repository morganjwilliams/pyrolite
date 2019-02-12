import numpy as np
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def FeAt8MgO(FEOT: float, MGO: float) -> float:
    """
    To account for differences in the slopes and curvature of liquid lines of descent as a function of parental magma composition.

    From Castillo2000
    DOI: 10.1029/1999GC000024
    http://onlinelibrary.wiley.com/doi/10.1029/1999GC000024/full
    Modified after Klein1987
    DOI: 10.1029/JB092iB08p08089
    http://onlinelibrary.wiley.com/doi/10.1029/JB092iB08p08089/full
    Similar to Langmuir1984
    """
    Fe8 = 1.825 - 1.529 * (FEOT - 0.03261 * MGO ** 2 + 0.2619) / (
        MGO - 0.04467 * MGO ** 2 - 6.67
    )
    return Fe8


def NaAt8MgO(NA2O: float, MGO: float) -> float:
    """
    To account for differences in the slopes and curvature of liquid lines of descent as a function of parental magma composition.

    From Castillo2000
    DOI: 10.1029/1999GC000024
    http://onlinelibrary.wiley.com/doi/10.1029/1999GC000024/full
    Modified after Klein1987
    DOI: 10.1029/JB092iB08p08089
    http://onlinelibrary.wiley.com/doi/10.1029/JB092iB08p08089/full
    Similar to Langmuir1984
    """
    Na8 = 0.6074 - 3.523 * (NA2O + 0.00529 * MGO ** 2 - 0.9495) / (
        MGO - 0.05297 * MGO ** 2 - 8.133
    )
