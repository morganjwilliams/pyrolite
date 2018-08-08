import os, sys
import re
import pandas as pd
import logging
from .util.melts import *
from .util.pd import to_frame
from pyrolite.geochem import common_oxides, common_elements
from pyrolite.util.melts import MELTS_Env


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



def from_melts_cstr(composition_str):
    """Parses melts composition strings to dictionaries."""
    regex = r"""(?P<el>[a-zA-Z'^.]+)(?P<num>[^a-zA-Z]+)"""
    result = re.findall(regex, composition_str)
    convert_element = lambda s: re.sub(r"""[\']+""",
                                       str(s.count("""'"""))+'+',
                                       s)
    return {convert_element(el): float(val) for (el, val) in result}


def to_meltsfiles(df, linesep=os.linesep, **kwargs):
    """
    Creates a number of melts files from a dataframe.
    """

    # Type checking such that series will be passed directly to MELTSfiles
    if isinstance(df, pd.DataFrame):
        for ix in range(df.index.size):
            to_meltsfile(df.iloc[ix, :])
    elif isinstance(df, pd.Series):
        to_meltsfile(src, **kwargs)


def to_meltsfile(ser, linesep=os.linesep, **kwargs):
    lines = []
    # majors -->  SiO2 45.7
    # output majors to Wt% values, may need to reorder them for MELTS..
    majors = [i for i in ser.index if i in common_oxides()]
    for k, v in zip(majors, ser[majors].values):
        if not pd.isnull(v): # no NaN data in MELTS files
            lines.append('Initial Composition: {} {}'.format(k, v))

    # traces --> Initial Trace: Sm 0.2

    # output traces to ppm values
    traces = [i for i in ser.index if i in common_elements()]
    for k, v in zip(traces, ser[traces].values):
        if not pd.isnull(v): # no NaN data in MELTS files
            lines.append('Initial Trace: {} {}'.format(k, v))
    # output valid kwargs
    valid = ['Mode',
             'Temperature',
             'Pressure',
             'dp/dt',
             'log fo2 Path',
             'Log fO2 Delta']

    # potentially pass these as tuples (start, stop, increment)
    # temperature, pressure --> Initial Temperature: 1500.0
    # temperature, pressure --> Final Temperature: 2000.0
    # temperature, pressure --> Increment Temperature: 3.00

    # dp/dt: 0.00
    # log fo2 Path: None
    # Log fO2 Delta: 0.0
    # Mode: Fractionate Solids
    return linesep.join(lines)
