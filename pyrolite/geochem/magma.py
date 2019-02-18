import numpy as np
from ..util.meta import update_docstring_references
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

@update_docstring_references
def FeAt8MgO(FEOT: float, MGO: float) -> float:
    """
    To account for differences in the slopes and curvature of liquid lines of descent
    as a function of parental magma composition [#ref_1]_ [#ref_2]_.

    References
    -----------
    .. [#ref_1] Castillo PR, Klein E, Bender J, et al (2000).
        Petrology and Sr, Nd, and Pb isotope geochemistry of
        mid-ocean ridge basalt glasses from the 11°45’N to 15°00’N
        segment of the East Pacific Rise.
        Geochemistry, Geophysics, Geosystems 1:1.
        doi: `10.1029/1999GC000024 <https://dx.doi.org/10.1029/1999GC000024>`__

    .. [#ref_2] Klein EM, Langmuir CH (1987).
        Global correlations of ocean ridge basalt chemistry with
        axial depth and crustal thickness.
        Journal of Geophysical Research: Solid Earth 92:8089–8115.
        doi: `10.1029/JB092iB08p08089 <https://dx.doi.org/10.1029/JB092iB08p08089>`__

    .. [#ref_3] Langmuir CH, Bender JF (1984).
        The geochemistry of oceanic basalts in the vicinity
        of transform faults: Observations and implications.
        Earth and Planetary Science Letters 69:107–127.
        doi: `10.1016/0012-821X(84)90077-3 <https://dx.doi.org/10.1016/0012-821X(84)90077-3>`__
    """
    Fe8 = 1.825 - 1.529 * (FEOT - 0.03261 * MGO ** 2 + 0.2619) / (
        MGO - 0.04467 * MGO ** 2 - 6.67
    )
    return Fe8

@update_docstring_references
def NaAt8MgO(NA2O: float, MGO: float) -> float:
    """
    To account for differences in the slopes and curvature of liquid lines of descent
    as a function of parental magma composition [#ref_1]_ [#ref_2]_.

    References
    -----------
    .. [#ref_1] Castillo PR, Klein E, Bender J, et al (2000).
        Petrology and Sr, Nd, and Pb isotope geochemistry of
        mid-ocean ridge basalt glasses from the 11°45’N to 15°00’N
        segment of the East Pacific Rise.
        Geochemistry, Geophysics, Geosystems 1:1.
        doi: `10.1029/1999GC000024 <https://dx.doi.org/10.1029/1999GC000024>`__

    .. [#ref_2] Klein EM, Langmuir CH (1987).
        Global correlations of ocean ridge basalt chemistry with
        axial depth and crustal thickness.
        Journal of Geophysical Research: Solid Earth 92:8089–8115.
        doi: `10.1029/JB092iB08p08089 <https://dx.doi.org/10.1029/JB092iB08p08089>`__

    .. [#ref_3] Langmuir CH, Bender JF (1984).
        The geochemistry of oceanic basalts in the vicinity
        of transform faults: Observations and implications.
        Earth and Planetary Science Letters 69:107–127.
        doi: `10.1016/0012-821X(84)90077-3 <https://dx.doi.org/10.1016/0012-821X(84)90077-3>`__
    """
    Na8 = 0.6074 - 3.523 * (NA2O + 0.00529 * MGO ** 2 - 0.9495) / (
        MGO - 0.05297 * MGO ** 2 - 8.133
    )
