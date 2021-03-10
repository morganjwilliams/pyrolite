"""
Submodule for calculating and modelling melt chemistry. Includes common
functions for predicting and accounting for melt evolution.
"""
import numpy as np
import pandas as pd
import periodictable as pt
from .ind import __common_elements__, __common_oxides__
from .transform import to_molecular, to_weight
from ..util.meta import update_docstring_references
from ..util.units import scale
from ..util.log import Handle

logger = Handle(__name__)


@update_docstring_references
def FeAt8MgO(FeOT: float, MgO: float) -> float:
    """
    To account for differences in the slopes and curvature of liquid lines of descent
    as a function of parental magma composition [#ref_1]_ [#ref_2]_ (after [#ref_3]_).

    Parameters
    -------------
    FeOT : :class:`float`
        Iron oxide content.
    MgO : :class:`float`
        Magnesium oxide content.

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
    Fe8 = 1.825 - 1.529 * (FeOT - 0.03261 * MgO ** 2 + 0.2619) / (
        MgO - 0.04467 * MgO ** 2 - 6.67
    )
    return Fe8


@update_docstring_references
def NaAt8MgO(Na2O: float, MgO: float) -> float:
    """
    To account for differences in the slopes and curvature of liquid lines of descent
    as a function of parental magma composition [#ref_1]_ [#ref_2]_ (after [#ref_3]_).

    Parameters
    -------------
    Na2O : :class:`float`
        Iron oxide content.
    MgO : :class:`float`
        Magnesium oxide content.

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
    Na8 = 0.6074 - 3.523 * (Na2O + 0.00529 * MgO ** 2 - 0.9495) / (
        MgO - 0.05297 * MgO ** 2 - 8.133
    )
    return Na8


@update_docstring_references
def SCSS(df, T, P, kelvin=False, grid=None, outunit="wt%"):
    r"""
    Obtain the sulfur content at sulfate and sulfide saturation [#ref_1]_ [#ref_2]_.

    Parameters
    -------------
    df : :class:`pandas.DataFrame`
        Dataframe of compositions.
    T : :class:`float` | :class:`numpy.ndarray`
        Temperature
    P : :class:`float` | :class:`numpy.ndarray`
        Pressure (kbar)
    kelvin : :class:`bool`
        Whether temperature values are in kelvin (:code:`True`) or celsuis (:code:`False`)
    grid : :code:`None`, :code:`'geotherm'`, :code:`'grid'`
        Whether to consider temperature and pressure as a geotherm (:code:`geotherm`),
        or independently (as a grid, :code:`grid`).

    Returns
    -------
    sulfate, sulfide : :class:`numpy.ndarray`, :class:`numpy.ndarray`
        Arrays of mass fraction sulfate and sulfide abundances at saturation.

    Notes
    ------

    For anhydrite-saturated systems, the sulfur content at sulfate saturation is given
    by the following:

    .. math::

        \begin{align}
        ln(X_S) = &10.07
        - 1.151 \cdot (10^4 / T_K)
        + 0.104 \cdot P_{kbar}\\
        &- 7.1 \cdot X_{SiO_2}
        - 14.02 \cdot X_{MgO}
        - 14.164 \cdot X_{Al_2O_3}\\
        \end{align}

    For sulfide-liquid saturated systems, the sulfur content at sulfide saturation is
    given by the following:

    .. math::

        \begin{align}
        ln(X_S) = &{-1.76}
        - 0.474 \cdot (10^4 / T_K)
        + 0.021 \cdot P_{kbar}\\
        &+ 5.559 \cdot X_{FeO}
        + 2.565 \cdot X_{TiO_2}
        + 2.709 \cdot X_{CaO}\\
        &- 3.192 \cdot X_{SiO_2}
        - 3.049 \cdot X_{H_2O}\\
        \end{align}

    References
    -----------
    .. [#ref_1] Li, C., and Ripley, E.M. (2009).
        Sulfur Contents at Sulfide-Liquid or Anhydrite Saturation in Silicate Melts:
        Empirical Equations and Example Applications. Economic Geology 104, 405–412.
        doi: `gsecongeo.104.3.405 <https://doi.org/10.2113/gsecongeo.104.3.405>`__

    .. [#ref_2] Smythe, D.J., Wood, B.J., and Kiseeva, E.S. (2017).
        The S content of silicate melts at sulfide saturation:
        New experiments and a model incorporating the effects of sulfide composition.
        American Mineralogist 102, 795–803.
        doi: `10.2138/am-2017-5800CCBY <https://doi.org/10.2138/am-2017-5800CCBY>`__

    Todo
    -----
    * Produce an updated version based on log-regressions?
    * Add updates from Smythe et al. (2017)?
    """
    T, P = np.array(T, dtype="float"), np.array(P, dtype="float")
    if not kelvin:
        T = T + 273.15

    C = np.ones(df.index.size)
    assert grid in [None, "geotherm", "grid"]
    if grid == "grid":
        cc, tt, pp = np.meshgrid(C, T, P, indexing="ij")
    elif grid == "geotherm":
        assert T.shape == P.shape
        cc = C[:, np.newaxis]
        tt = T[np.newaxis, :]
        pp = P[np.newaxis, :]
    elif grid is None:
        _dims = C.size, T.size, P.size
        maxdim = max(_dims)
        assert all([x == maxdim or x == 1 for x in _dims])
        cc, tt, pp = C, T, P

    comp = set(df.columns) & (__common_elements__ | __common_oxides__)
    moldf = to_molecular(df.loc[:, comp], renorm=True) / 100.0  # mole-fraction
    molsum = to_molecular(df.loc[:, comp], renorm=False).sum(axis=1)

    def gridify(ser):
        """
        Create a parameter grid from a pandas series to facilitate
        array-addition of each component.
        """
        arr = ser.replace(np.nan, 0).values
        if grid == "grid":
            return arr[:, np.newaxis, np.newaxis]
        elif grid == "geotherm":
            return arr[:, np.newaxis]
        elif grid is None:
            return arr

    ln_sulfate = 10.07 * cc - 1.151 * 10 ** 4 / tt + 0.104 * pp
    for chem, D in [("SiO2", -7.1), ("MgO", -14.02), ("Al2O3", -14.164)]:
        if chem in moldf.columns:
            ln_sulfate += gridify(moldf[chem]) * D * cc

    ln_sulfide = -1.76 * cc - 0.474 * 10 ** 4 / tt + 0.021 * pp

    for chem, D in [
        ("FeO", 5.559),
        ("TiO2", 2.565),
        ("CaO", 2.709),
        ("SiO2", -3.192),
        ("H2O", -3.049),
    ]:
        if chem in moldf.columns:
            ln_sulfide += gridify(moldf[chem]) * D * cc

    sulfate, sulfide = np.exp(ln_sulfate), np.exp(ln_sulfide)

    _s = gridify(molsum * pt.S.mass) * scale("wt%", outunit)
    _so4 = gridify(molsum * pt.formula("SO4").mass) * scale("wt%", outunit)
    sulfide *= _s
    sulfate *= _so4

    if sulfate.size == 1:  # 0D
        return sulfate.flatten()[0], sulfide.flatten()[0]
    else:  # 2D
        return sulfate, sulfide
