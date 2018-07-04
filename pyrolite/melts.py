import sys
import platform
import os
import subprocess
from environs import Env
import json
import requests
import dicttoxml
import re
from xml.dom.minidom import parseString

def urlify(s):

     # Remove all non-word characters (everything except numbers and letters)
     s = re.sub(r"""[\n]""", '', s)

     # Replace all runs of whitespace with a single dash
     s = re.sub(r"\s+", '', s)

     return s

def melts_webquery():

    default_data = dict(MELTSinput = dict(
                            initialize={"SiO2": "48.68",
                                        "TiO2": "1.01",
                                        "Al2O3": "17.64",
                                        "Fe2O3": "0.89",
                                        "Cr2O3": "0.0425",
                                        "FeO": "7.59",
                                        "MnO": "0.0",
                                        "MgO": "9.10",
                                        "NiO": "0.0",
                                        "CoO": "0.0",
                                        "CaO": "12.45",
                                        "Na2O": "2.65",
                                        "K2O": "0.03",
                                        "P2O5": "0.08",
                                        "H2O": "0.20" },
                            calculationMode='equilibrate',
                            title='TestREST',
                            constraints={"setTP": {"initialT": "1200",
                                                   "initialP": "1000" }
                                                   })
                        )
    dataxml = dicttoxml.dicttoxml(default_data,
                                  #custom_root='MELTSinput',
                                  root=True,
                                  attr_type=False
                                  )#.decode("UTF-8")
    dom = parseString(dataxml)
    print(dom.toprettyxml())
    payload = dict(data=dataxml,
                   contentType="text/xml",
                   dataType="xml")

    url = "http://thermofit.ofm-research.org:8080/multiMELTSWSBxApp/Compute"
    resp = requests.post(url, params=payload)
    print(resp.url)
    resp.raise_for_status()
    raw = resp.content
    print(raw)

melts_webquery()

def MELTS_env():
    env = Env()

    with env.prefixed('ALPHAMELTS_'):

        version = env.str('VERSION', 'pMELTS') # MELTS, pMELTS
        mode = env.str('MODE', 'isentropic')  # ‘geothermal’, ‘isenthalpic’, ‘isentropic’, ‘isobaric’, ‘isochoric’, ‘isothermal’, ‘PTpath’

        _maxP = [30000, 40000][env('VERSION') != 'MELTS']  # in degC
        _minP = [1, 10000][env('VERSION') != 'MELTS']  # in degC
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

    env.dump()

MELTS_env()

def locate_melts():

    system = platform.system()
    release = platform.release()

    if system =='Linux':
        pass
    elif system == 'Darwin':
        pass
    elif system == 'Windows':
        pass
    else:
        raise NotImplementedError(f'System unknown: {system}')

    return system


def install_melts(directory):


    # Set install directory for .bat files
    install_dir = '.'

    # Export links

    # export to path
    if install_dir not in sys.path:
        sys.path.append(install_dir)

    p = subprocess.run([exe_path],
                       input=stdin,
                       stdout=subprocess.PIPE,
                       universal_newlines=True)


locate_melts()
    #if sys.platform


def parse_melts_mineral_composition(composition_str):

    """e.g. for spinel: Fe''0.18Mg0.83Fe'''0.04Al1.43Cr0.52Ti0.01O4"""

    """Fe''0.18Mg0.83Fe'''0.04Al1.43Cr0.52Ti0.01O4"""



def amoeba():

    # read melts file

    # set ALPHAMELTS_FRACTIONATE_TARGET
    subprocess.run()
