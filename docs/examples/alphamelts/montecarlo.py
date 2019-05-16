import numpy as np
import pandas as pd

# %% setup an environment for isenthalpic fractional crystallisation
from pyrolite.util.alphamelts.env import MELTS_Env

env = MELTS_Env()
env.VERSION = "MELTS"  # crustal processes, pMELTS > 1GPA/10kbar
env.MODE = "isenthalpic"
env.DELTAT = -5
env.DELTAP = 0
env.MINP = 100
env.MAXP = 10000
env.MINT = 800
env.MAXT = 1800

# %% get the MORB melts file
from pyrolite.geochem.norm import ReferenceCompositions
from pyrolite.geochem.ind import __common_oxides__
from pyrolite.comp.codata import renormalise, ilr, inverse_ilr
from pyrolite.util.pd import accumulate, to_numeric
from pyrolite.util.alphamelts.meltsfile import to_meltsfile

# get the major element composition of MORB from Gale et al (2013)
Gale_MORB = ReferenceCompositions()["MORB_Gale2013"]
MORB = Gale_MORB.original_data.loc[
    ["SiO2", "Al2O3", "FeO", "MnO", "MgO", "CaO", "Na2O", "TiO2", "K2O", "P2O5"],
    "value",
]
MORB["Title"] = Gale_MORB.ModelName
MORB["Initial Temperature"] = 1300
MORB["Final Temperature"] = 800
MORB["Initial Pressure"] = 5000
MORB["Final Pressure"] = 5000
MORB["Log fO2 Path"] = "FMQ"
MORB["Increment Temperature"] = -5
MORB["Increment Pressure"] = 0
# %% replicate and add noise

def blur_compositions(df, noise=0.05, scale=100):
    """
    Function to add 'compositional noise' to a set of compositions. In reality, it's
    its best to use measured uncertainties to generate these simulated compositions.
    """
    # transform into compositional space, add noise, return to simplex
    xvals = ilr(df.values)
    xvals += np.random.randn(*xvals.shape) * noise
    return inverse_ilr(xvals) * scale


replicates = 50
meltsfiles = (
    accumulate([pd.DataFrame(MORB).T] * replicates).reset_index().drop(columns="index")
)

compositional_vars = [i for i in meltsfiles if i in __common_oxides__]
meltsfiles[compositional_vars] = to_numeric(meltsfiles[compositional_vars])
meltsfiles[compositional_vars] = (
    meltsfiles[compositional_vars].astype(float).renormalise()
)

meltsfiles[compositional_vars] = blur_compositions(meltsfiles[compositional_vars])
# %% run the models for each of the inputs
from pyrolite.util.general import temp_path
from pyrolite.util.text import slugify
from pyrolite.util.alphamelts.automation import MeltsExperiment

tempdir = temp_path() / "test_temp_montecarlo"

# differentiate titles
meltsfiles.Title = meltsfiles.Title + " " + meltsfiles.index.map(str)
meltsfiles.Title = meltsfiles.Title.apply(slugify)

for ix in meltsfiles.index:
    title = meltsfiles.loc[ix, "Title"]
    meltsfile = to_meltsfile(
        meltsfiles.loc[ix, :],  # take the specific row
        writetraces=False,  # ignore trace element data
        modes=["isenthalpic"],  # conduct an isenthalpic experiment
        exclude=["P2O5", "K2O"],  # leave out the K2O and P2O5
    )
    exp = MeltsExperiment(meltsfile=meltsfile, title=title, env=env, dir=tempdir)
    exp.run(superliquidus_start=True)

# %% aggregate the results over the same gridded space (realising that some may fail differently)
from pyrolite.util.alphamelts.tables import get_experiments_summary
from pyrolite.util.alphamelts.plottemplates import table_by_phase

summary = get_experiments_summary(tempdir, kelvin=False)

fig = table_by_phase(summary)

# %% save figure
from pyrolite.util.plot import save_figure

save_figure(fig, save_at="../../source/_static", name="melts_montecarlo")
