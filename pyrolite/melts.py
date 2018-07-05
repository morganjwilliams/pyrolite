import os, sys
import re
import platform
import subprocess
import requests
from xml.etree import ElementTree as ET
import xmljson
import dicttoxml
from environs import Env


default_data = dict(initialize={"SiO2": 48.68,
                                    "TiO2": 1.01,
                                    "Al2O3": 17.64,
                                    "Fe2O3": 0.89,
                                    "Cr2O3": 0.0425,
                                    "FeO": 7.59,
                                    "MnO": 0.0,
                                    "MgO": 9.10,
                                    "NiO": 0.0,
                                    "CoO": 0.0,
                                    "CaO": 12.45,
                                    "Na2O": 2.65,
                                    "K2O": 0.03,
                                    "P2O5": 0.08,
                                    "H2O": 0.20 },
                        calculationMode='findLiquidus',
                        title='TestREST',
                        constraints={"setTP": {"initialT": 1200,
                                               "initialP": 1000}
                                               }
                    )



def melts_query(data_dict, url_sfx='Compute'):
    url = 'http://thermofit.ofm-research.org:8080/multiMELTSWSBxApp/' + url_sfx
    xmldata = dicttoxml.dicttoxml(data_dict,
                                  custom_root='MELTSinput',
                                  root=True,
                                  attr_type=False)
    headers = {"content-type": "text/xml",
               "data-type": "xml"}
    resp = requests.post(url, data=xmldata, headers=headers)
    resp.raise_for_status()
    result = xmljson.parker.data(ET.fromstring(resp.text))
    return result


def melts_compute(data_dict):
    url_sfx = "Compute"
    result = melts_query(data_dict, url_sfx=url_sfx)
    assert 'Success' in result['status']
    return result


def melts_oxides(data_dict):
    model = data_dict['initialize'].pop('modelSelection', 'MELTS_v1.0.x')
    data_dict = {'modelSelection': model}
    url_sfx = "Oxides"
    result = melts_query(data_dict, url_sfx=url_sfx)
    return result['Oxide']


def melts_phases(data_dict):
    model = data_dict['initialize'].pop('modelSelection', 'MELTS_v1.0.x')
    data_dict = {'modelSelection': model}
    url_sfx = "Phases"
    result = melts_query(data_dict, url_sfx=url_sfx)
    return result['Phase']


melts_phases(default_data)
ret = melts_query(default_data)

ret

class MeltsSystem:

    def __init__(self, composition):

        self.composition = composition
        self.liquid = None
        self.solid = None
        self.potentialSolid = None
        self.parameters = None

    def equilirate(self):
        method = 'equilirate'

    def findLiquidus(self):
        method = 'findLiquidus'

    def findWetLiquidus(self):
        method = 'findWetLiquidus'


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
