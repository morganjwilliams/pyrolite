import matplotlib.pyplot as plt
from pyrolite.geochem.ind import __common_oxides__
from pyrolite.ext.alphamelts.tables import MeltsOutput

# %% ---
# say you have a folder containing melts tables
output = MeltsOutput(folder, kelvin=False)  # tables in degrees C

# this object has a number of useful attributes

output.tables  # list of tables accessible from the object

{"bulkcomp", "liquidcomp", "phasemass", "phasevol", "solidcomp", "system", "tracecomp"}

output.phasenames  # get the names of phases which appear in the experiment

{"clinopyroxene_0", "feldspar_0", "liquid_0", "olivine_0", "spinel_0"}

output.phases  # dictionary of phasename : phase composition tables (<df>)

{
    "liquid_0": "<df>",
    "spinel_0": "<df>",
    "feldspar_0": "<df>",
    "clinopyroxene_0": "<df>",
    "olivine_0": "<df>",
}

# e.g. access the phase Volume table using:
output.phasevol
