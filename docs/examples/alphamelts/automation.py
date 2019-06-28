import pandas as pd
from pyrolite.ext.alphamelts.meltsfile import to_meltsfile
from pyrolite.ext.alphamelts.automation import MeltsExperiment, make_meltsfolder
# %% testdf
import numpy as np
from pyrolite.util.synthetic import test_df

df = test_df(cols=["SiO2", "CaO", "MgO", "FeO", "TiO2", "Na2O", "K2O", "P2O5"]) * 100
df["Sample"] = np.arange(df.index.size)
# %% setup environment
from pyrolite.ext.alphamelts.env import MELTS_Env

env = MELTS_Env()
env.VERSION = "MELTS"  # crustal processes, pMELTS > 1GPA/10kbar
env.MODE = "isobaric"
env.MINT = 700
env.MINP = 2000
env.DELTAT = -3

with open("pyrolite_envfile.txt", "w") as f:  # write the environment to a file
    f.write(env.to_envfile(unset_variables=False))
# %% setup dataframe
# taking a dataframe with oxide/element headers, set up experiment info
df["Title"] = df.Sample
df["Initial Pressure"] = 7000
df["Initial Temperature"] = 1350
df["Final Temperature"] = 1000
df["Increment Temperature"] = -3
df["Log fO2 Path"] = "FMQ"
# %% autorun
# we can create melts files from each of these data rows, and run an automated experiment
for ix in df.index:
    meltsfile = to_meltsfile(
        df.loc[ix, :],  # take the specific row
        writetraces=False,  # ignore trace element data
        modes=[
            "isobaric",
            "fractionate solids",
        ],  # conduct an isobaric experiment where solids are fractionated
        exclude=["P2O5", "K2O"],  # exclude potassium and phosphorous
    )
    # create an experiment folder to work in, add the meltsfile and environment file
    exp = MeltsExperiment(meltsfile=meltsfile, title=str(df.loc[ix, "Title"]), env=env)
    exp.run(superliquidus_start=True)
