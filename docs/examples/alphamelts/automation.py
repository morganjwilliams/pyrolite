import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pyrolite.plot
from pyrolite.geochem.norm import get_reference_composition
from pyrolite.ext.alphamelts.env import MELTS_Env
from pyrolite.ext.alphamelts.automation import MeltsBatch

from pyrolite.util.meta import stream_log
import logging

logger = logging.Logger(__name__)
stream_log(logger) # print the logging output
# %% Data
Gale_MORB = get_reference_composition("MORB_Gale2013")
majors = ["SiO2", "Al2O3", "FeO", "MnO", "MgO", "CaO", "Na2O", "TiO2", "K2O", "P2O5"]
MORB = Gale_MORB[majors]
MORB["Title"] = [
    "{}_{}".format(Gale_MORB.name, ix)
    for ix in MORB.index.values.astype(str)
]
MORB
MORB["Initial Temperature"] = 1300
MORB["Final Temperature"] = 800
MORB["Initial Pressure"] = 5000
MORB["Final Pressure"] = 5000
MORB["Log fO2 Path"] = "FMQ"
MORB["Increment Temperature"] = -5
MORB["Increment Pressure"] = 0
# %% Environment
env = MELTS_Env()
env.VERSION = "MELTS"
env.MODE = "isobaric"
env.MINP = 5000
env.MAXP = 10000
env.MINT = 500
env.MAXT = 1500
env.DELTAT = -10
env.DELTAP = 0
# %% Batch
batch = MeltsBatch(
    MORB,
    default_config={
        "Initial Pressure": 7000,
        "Initial Temperature": 1400,
        "Final Temperature": 800,
        "modes": ["isobaric", "fractionate solids"],
    },
    grid={
        "Initial Pressure": [5000],
        "Log fO2 Path": [None, "FMQ"],
        "modifychem": [None, {"H2O": 0.5}],
    },
    env=env,
    logger=logger,
)

batch.run(overwrite=True) # overwrite=False if you don't want to update existing exp folders
