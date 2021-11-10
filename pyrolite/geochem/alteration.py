"""
Functions for calcuating indexes of chemical alteration.
"""
import numpy as np
import pandas as pd

from ..util.log import Handle
from ..util.meta import update_docstring_references

logger = Handle(__name__)


@update_docstring_references
def CIA(df: pd.DataFrame):
    """
    Chemical Index of Alteration (molecular) [#ref_1]_

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        DataFrame to calculate index from.

    Returns
    --------
    :class:`pandas.Series`
        Alteration index series.

    References
    ----------
    .. [#ref_1] Nesbitt HW, Young GM (1984). Prediction of some weathering trends of plutonic
           and volcanic rocks based on thermodynamic and kinetic considerations.
           Geochimica et Cosmochimica Acta 48:1523–1534.
           doi: `10.1016/0016-7037(84)90408-3 <https://dx.doi.org/10.1016/0016-7037(84)90408-3>`__

    """
    return 100 * df.Al2O3 / (df.Al2O3 + df.CaO + df.Na2O + df.K2O)


@update_docstring_references
def CIW(df: pd.DataFrame):
    """
    Chemical Index of Weathering (molecular) [#ref_1]_

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        DataFrame to calculate index from.

    Returns
    --------
    :class:`pandas.Series`
        Alteration index series.

    References
    ----------
    .. [#ref_1] Harnois L (1988). The CIW index: A new chemical index of weathering.
           Sedimentary Geology 55:319–322. doi:
           `10.1016/0037-0738(88)90137-6 <https://dx.doi.org/10.1016/0037-0738(88)90137-6>`__
    """
    return 100.0 * df.Al2O3 / (df.Al2O3 + df.CaO + df.Na2O)


@update_docstring_references
def PIA(df: pd.DataFrame):
    """
    Plagioclase Index of Alteration (molecular) [#ref_1]_

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        DataFrame to calculate index from.

    Returns
    --------
    :class:`pandas.Series`
        Alteration index series.

    References
    ----------
    .. [#ref_1] Fedo CM, Nesbitt HW, Young GM (1995).
           Unraveling the effects of potassium metasomatism
           in sedimentary rocks and paleosols, with implications for paleoweathering
           conditions and provenance. Geology 23:921–924.
           doi: `10.1130/0091-7613(1995)023<0921:UTEOPM>2.3.CO;2 <https://dx.doi.org/10.1130/0091-7613(1995)023<0921:UTEOPM>2.3.CO;2>`__

    """
    return 100.0 * (df.Al2O3 - df.K2O) / (df.Al2O3 + df.CaO + df.Na2O - df.K2O)


@update_docstring_references
def SAR(df: pd.DataFrame):
    """
    Silica-Alumina Ratio (molecular)

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        DataFrame to calculate index from.

    Returns
    --------
    :class:`pandas.Series`
        Alteration index series.
    """
    return df.SiO2 / df.Al2O3


@update_docstring_references
def SiTiIndex(df: pd.DataFrame):
    """
    Silica-Titania Index (molecular) [#ref_1]_

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        DataFrame to calculate index from.

    Returns
    --------
    :class:`pandas.Series`
        Alteration index series.

    References
    ----------
    .. [#ref_1] Jayawardena U de S, Izawa E (1994).
            A new chemical index of weathering for metamorphic silicate rocks in
            tropical regions: A study from Sri Lanka.
            Engineering Geology 36:303–310.
            doi: `10.1016/0013-7952(94)90011-6 <https://dx.doi.org/10.1016/0013-7952(94)90011-6>`__
    """
    # may need to recalculate titania from titanium ppm
    si_ti = df.SiO2 / df.TiO2
    si_al = df.SiO2 / df.Al2O3
    al_ti = df.Al2O3 / df.TiO2
    return si_ti / (si_ti + si_al + al_ti)


@update_docstring_references
def WIP(df: pd.DataFrame):
    """
    Weathering Index of Parker (molecular) [#ref_1]_

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        DataFrame to calculate index from.

    Returns
    --------
    :class:`pandas.Series`
        Alteration index series.

    References
    ----------
    .. [#ref_1] Parker A (1970). An Index of Weathering for Silicate Rocks.
           Geological Magazine 107:501–504.
           doi: `10.1017/S0016756800058581 <https://dx.doi.org/10.1017/S0016756800058581>`__

    """
    return 2 * df.Na2O / 0.35 + df.MgO / 0.9 + 2 * df.K2O / 0.25 + df.CaO / 0.7


@update_docstring_references
def IshikawaAltIndex(df: pd.DataFrame):
    """
    Alteration Index of Ishikawa (wt%) [#ref_1]_

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        DataFrame to calculate index from.

    Returns
    --------
    :class:`pandas.Series`
        Alteration index series.

    References
    ----------
    .. [#ref_1] Ishikawa Y, Sawaguchi T, Iwaya S, Horiuchi M (1976).
            Delineation of prospecting targets for Kuroko deposits based on
            modes of volcanism of underlying dacite and alteration halos.
            Mining Geology 27:106-117.

    """
    return 100 * (df.K2O + df.MgO) / (df.K2O + df.MgO + df.Na2O + df.CaO)


@update_docstring_references
def CCPI(df: pd.DataFrame):
    """
    Chlorite-carbonate-pyrite index of Large et al. (wt%) [#ref_1]_.


    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        DataFrame to calculate index from.

    Returns
    --------
    :class:`pandas.Series`
        Alteration index series.

    Notes
    -----
    Here FeO is taken as total iron (FeO+Fe2O3).

    References
    ----------
    .. [#ref_1] Large RR, Gemmell, JB, Paulick, H (2001). The alteration box
           plot: A simple approach to understanding the relationship between
           alteration mineralogy and lithogeochemistry associated with
           volcanic-hosted massive sulfide deposits.
           Economic Geology 96:957-971.
           doi:`10.2113/gsecongeo.96.5.957 <http://dx.doi.org/10.2113/gsecongeo.96.5.957>`__

    """
    if "FeO" in df.columns:
        ccpi = 100 * (df.MgO + df.FeO) / (df.MgO + df.FeO + df.Na2O + df.K2O)
    elif "Fe2O3" in df.columns and "FeO" in df.columns:
        ccpi = (
            100
            * (df.MgO + df.FeO + df.Fe2O3)
            / (df.MgO + df.FeO + df.Fe2O3 + df.Na2O + df.K2O)
        )
    return ccpi
