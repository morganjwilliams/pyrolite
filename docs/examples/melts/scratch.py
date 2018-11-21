
from collections import OrderedDict
from pyrolite.melts import *
from pyrolite.util.melts import *

def get_default_datadict():
    d = OrderedDict()
    d['title'] = 'TestREST',
    d['initialize'] = {"SiO2": 48.68,
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
                        "H2O": 0.20 }
    d['calculationMode']='findLiquidus'
    d['constraints'] = {"setTP": {"initialT": 1200,
                                  "initialP": 1000}}
    return d

D = get_default_datadict()

# liquid, system, potentialSolid

info = melts_compute(D)
#print(info.keys())

class Component(object):

    def __init__(self, *args, **kwargs):

        pass

import pandas as pd
d = pd.DataFrame()
d[0] = pd.Series({'title': D['title'], **D['initialize'], **D['constraints']['setTP']})
type(d[0])


def to_meltsfile(ser, linesep=os.linesep, **kwargs):
    """
    Converts a series to a MELTSfile text representation.
    """
    lines = []
    # majors -->  SiO2 45.7
    ser = to_ser(ser)
    # output majors to Wt% values, may need to reorder them for MELTS..
    majors = [i for i in ser.index if i in common_oxides()]
    for k, v in zip(majors, ser.loc[majors].values):
        if not pd.isnull(v): # no NaN data in MELTS files
            lines.append('Initial Composition: {} {}'.format(k, v))

    # traces --> Initial Trace: Sm 0.2

    # output traces to ppm values
    traces = [i for i in ser.index if i in common_elements()]
    for k, v in zip(traces, df.loc[:, traces].values):
        if not pd.isnull(v.any()): # no NaN data in MELTS files
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

for l in to_meltsfile(d[0]).split('\n'):
    print(l)

env = MELTS_Env()
run_wds_command('alphamelts')

#%%------------------------------

import subprocess, os
from subprocess import PIPE

def to_cmd(inputs):
    if isinstance(inputs, str):
        return (inputs+'\n').encode('UTF-8')
    else:
        return ('\n'.join(inputs)+'\n').encode('UTF-8')

def _read_file(filename):
    return ['1', '', filename, '']

def _change_parameters(temperature=None, pressure=None):
    _t = [str([temperature, 0][temperature is None]), '']
    _p = [str([pressure, 0][pressure is None]), '']
    return ['2', ''] + _t + _p

def _single_calculation(phases=[],
                        subsolidus_guess:bool=True):
    """
    Liquidus calculation. Phases will be required if not set.
    """
    guess = str(int(not subsolidus_guess))

    _phx = []
    if phases:
        _phx += phases
        _phx += ['x']
    return ['3'] + [guess] + _phx

def _execute(subsolidus_guess:bool=True):
    """
    Liquidus calculation. Phases will be required if not set.
    """
    guess = str(int(not subsolidus_guess))

    _phx = []
    if phases:
        _phx += phases
        _phx += ['x']
    return ['4', guess] + _phx

def _set_fO2(buffer=None,
             offset=0):
    """
    Set the fO2 or aFe.
    """

    buffers = dict(None=0,
                   HM=1,
                   NNO=2,
                   QFM=3,
                   IW=4)

    buffer_id = buffers[buffer]
    return ['5', buffer_id, offset]

def _set_aH2O(a=0):
    """
    Set the a(H2O)
    """
    return ['6', a]

def _set_SHV(S=None, H=None, V=None,
             isothermal=False):
    """
    Set the initial entropy#, enthalpy or volume.?
    """
    if S is None:
        return []
    else:
        return ['6', [S, 0][isothermal]]

def _adjust_solids(phases=[],
                   settings=[]):
    """Adjust the settings for a solid phase."""
    phase_settings = zip(phases, settings)
    _phx = []
    if phases:
        for p, s in phase_settings:
            _phx += [p, s]
        _phx += ['x']
    return ['8'] + _phx


def _exit():
    return ['0', '']

args = ['cmd']
inputs = ["alphamelts", '',] + \
         _read_file('C:/melts/examples/Morb.melts') + \
         _change_parameters(temperature=1500) + \
         _single_calculation(subsolidus_guess=False) + \
         _exit()
p = subprocess.run(args, input=to_cmd(inputs), stdout=PIPE, timeout=10)

output=False
if output:
    for line in p.stdout.decode('UTF-8').split('\r\n'):
        print(line)

#%%

import subprocess, os
from subprocess import PIPE

def to_cmd(inputs):
    if isinstance(inputs, str):
        return (inputs+'\n').encode('UTF-8')
    else:
        return ('\n'.join(inputs)+'\n').encode('UTF-8')

p = subprocess.Popen('cmd', stdout=PIPE, stdin=PIPE)
stdin, stdout = p.stdin, p.stdout
args = ["alpahmelts", '0']
for a in args:
    stdin.write(to_cmd(a))
    while p.returncode is None:
        p.poll()

stdout.read()
#p.terminate()

#%%


outs, errs = p.communicate('', timeout=10)
outs.decode('UTF-8').split('\r\n')
outs, errs = p.communicate(list_to_args(["C;/melts/alphamelts_win64.exe"]), timeout=10)
outs.decode('UTF-8').split('\r\n')
p.terminate()

try:
    for i in inputs:
        outs, errs = p.communicate(input=(i+'\n').encode('UTF-8'),
                                               timeout=10)
        for _l in outs.decode('UTF-8').split('\r\n'):
            print(_l)
except:
finally:
    try:
        p.terminate()
        outs, errs = p.communicate()
    except:
        p.kill()


assert p.returncode == 0
