import numpy as np
import pandas as pd
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

def CIA(df:pd.DataFrame):
    """
    Chemical Index of Alteration
    type:   molecular
    ref:    Nesbitt and Young (1982)
    """
    return 100 * df.Al2O3 / (df.Al2O3 + df.CaO + df.Na2O + df.K2O)


def CIW(df:pd.DataFrame):
    """
    Chemical Index of Weathering
    type:   molecular
    ref:    Harnois (1988)
    """
    return 100. * df.Al2O3 / (df.Al2O3 + df.CaO + df.Na2O)


def PIA(df:pd.DataFrame):
    """
    Plagioclase Index of Alteration
    type:   molecular
    ref:    Fedo et al. (1995)
    """
    return 100. * (df.Al2O3 - df.K2O) / (df.Al2O3 + df.CaO + df.Na2O - df.K2O)


def SAR(df:pd.DataFrame):
    """
    Silica-Alumina Ratio
    type:   molecular
    ref:
    """
    return df.SiO2/df.Al2O3


def SiTiIndex(df:pd.DataFrame):
    """
    Silica-Titania Index
    Jayaverdena and Izawa (1994)
    type:   molecular
    ref:
    """
    # may need to recalculate titania from titanium ppm
    si_ti = df.SiO2/df.TiO2
    si_al = df.SiO2/df.Al2O3
    al_ti = df.Al2O3/df.TiO2
    return si_ti / (si_ti + si_al + al_ti)


def WIP(df:pd.DataFrame):
    """
    Weathering Index of Parker
    Parker (1970)
    type:   molecular
    ref:
    """
    return 2 * df.Na2O / 0.35  + df.MgO / 0.9 + \
           2 * df.K2O / 0.25 + df.CaO / 0.7
