import pandas as pd
from pyrolite.util.alphamelts.meltsfile import to_meltsfile
from pyrolite.util.alphamelts.automation import MeltsProcess, make_meltsfolder
# %% testdf
import numpy as np
from pyrolite.util.synthetic import test_df

df = test_df(cols=["SiO2", "CaO", "MgO", "FeO", "TiO2", "Na2O", "K2O", "P2O5"])
df["Sample"] = np.arange(df.index.size)
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
    meltsfile = to_meltsfile(  # write
        df.loc[ix, :],  # take the specific row
        writetraces=False,  # ignore trace element data
        modes=[
            "isobaric",
            "fractionate solids",
        ],  # conduct an isobaric experiment where solids are fractionated
        exclude=["P2O5", "K2O"],  # exclude potassium and phosphorous
    )
    # create an experiment folder to work in, add the meltsfile and environment file
    experiment_folder = make_meltsfolder(
        meltsfile, df.loc[ix, "Title"], env="pyrolite_envfile.txt"
    )
    # set up a melts process to run this automatically
    process = MeltsProcess(
        meltsfile=df.loc[ix, "Title"] + ".melts",  #
        env="pyrolite_envfile.txt",
        fromdir=str(experiment_folder),
    )
    # run some commands as you would in alphaMELTS,
    # if it errors all hope is lost (for now), and you'll have to run it manually
    process.write(3, 1, 4, wait=True, log=False)
    # end the experiment
    process.terminate()
