import os, sys
import re
import pandas as pd
import logging
from .util.melts import *

from .util.pd import to_frame


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


class MeltsSystem:

    def __init__(self, composition):

        self.composition = composition
        self.liquid = None
        self.solid = None
        self.potentialSolid = None
        self.parameters = None

    def equilirate(self):
        method = 'equilibrate'

    def findLiquidus(self):
        method = 'findLiquidus'

    def findWetLiquidus(self):
        method = 'findWetLiquidus'


def MELTS_env():
    env = Env()

    with env.prefixed('ALPHAMELTS_'):

        version = env.str('VERSION', 'pMELTS') # MELTS, pMELTS
        mode = env.str('MODE', 'isentropic')  # ‘geothermal’, ‘isenthalpic’, ‘isentropic’, ‘isobaric’, ‘isochoric’, ‘isothermal’, ‘PTpath’

        _maxP = [30000, 40000][version != 'MELTS']  # in degC
        _minP = [1, 10000][version != 'MELTS']  # in degC
        min_P = env('MINP', _minP)
        max_P = env.float('MAXP', _maxP)

        min_T = env('MINP', 0)  # in degC
        max_T = env.float('MAXP', 2000)  # in degC

        delta_P = env.float('DELTAP', '1000')  # in bars
        delta_T = env.float('DELTAT', '10')  # in degC

        # ptpath_file = env.str('PTPATH_FILE')
        # alt_FO2 = env('ALTERNATIVE_FO2')
        # liq_FO2 = env('LIQUID_FO2')
        # impose_FO2 = env('IMPOSE_FO2')
        # FO2_pressure_term = env('FO2_PRESSURE_TERM')
        # metastable_isograd = env('METASTABLE_ISOGRAD')


        # continuous_melting = env('CONTINUOUS_MELTING')
        """
        The following variables overwrite each other,
        with those further along taking precedence.
        """
        minf = env.float('MINF', 0.005)  # 0 < minf < 1
        # minphi = env.float('MINPHI')  # 0 < minphi < 1
        # continous_ratio = env.float('CONTINUOUS_RATIO')

        # continous_volume = env.float('CONTINUOUS_VOLUME')

        # fractionate_solids = env('FRACTIONATE_SOLIDS')
        # mass_in = env.float('MASSIN', 0.001)  # 0 < mass_in < 1
        # fractionate_water = env('FRACTIONATE_WATER')
        # min_w = env.float('MINW')  # 0 < minw < 1, overrides mass_in

        # fractionate_target = env('FRACTIONATE_TARGET')
        # MgO_target = env.float('MGO_TARGET', 8.0)
        # MgNo_target = env.float('MGNUMBER_TARGET')  #overrides MgO_target

        # assimilate = env('ASSIMILATE')

        # use_old_garnet = env('OLD_GARNET')  #any
        # use_old_biotite = env('OLD_BIOTITE')  # any
        # use_am2_amph = env('2_AMPH')  # any

    return env.dump()


def from_melts_cstr(composition_str):
    """Parses melts composition strings to dictionaries."""
    regex = r"""(?P<el>[a-zA-Z'^.]+)(?P<num>[^a-zA-Z]+)"""
    result = re.findall(regex, composition_str)
    convert_element = lambda s: re.sub(r"""[\']+""",
                                       str(s.count("""'"""))+'+',
                                       s)
    return {convert_element(el): float(val) for (el, val) in result}


def to_meltsfile(src):

    if isinstance(src, pd.DataFrame):
        pass
    elif isinstance(src, pd.Series):
        src = to_frame(src)


    'Initial'
    # majors --> Initial Composition: SiO2 45.7
    'Composition'
    # traces --> Initial Trace: Sm 0.2
    'Trace'
    # temperature, pressure --> Initial Temperature: 1500.0
    # temperature, pressure --> Final Temperature: 2000.0
    # temperature, pressure --> Increment Temperature: 3.00
    # dp/dt: 0.00
    # log fo2 Path: None
    # Log fO2 Delta: 0.0
    # Mode: Fractionate Solids
    pass
