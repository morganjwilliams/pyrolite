import numpy as np
import pandas as pd
from pathlib import Path
import pyrolite.geochem
import pyrolite.plot
from pyrolite.comp.codata import ilr, inverse_ilr
from pyrolite.util.meta import stream_log
import logging

logger = logging.Logger(__name__)
stream_log(logger)  # print the logging output


def blur_compositions(df, noise=0.05, scale=100):
    """
    Function to add 'compositional noise' to a set of compositions. In reality, it's
    its best to use measured uncertainties to generate these simulated compositions.
    """
    # transform into compositional space, add noise, return to simplex
    xvals = ilr(df.values)
    xvals += np.random.randn(*xvals.shape) * noise
    return inverse_ilr(xvals) * scale


# %% Data
from pyrolite.geochem.norm import get_reference_composition

# get the major element composition of MORB from Gale et al (2013)
Gale_MORB = get_reference_composition(["MORB_Gale2013"])
MORB = Gale_MORB.comp[majors].reset_index(drop=True)

MORB["Title"] = Gale_MORB.ModelName
MORB["Initial Temperature"] = 1300
MORB["Final Temperature"] = 800
MORB["Initial Pressure"] = 5000
MORB["Final Pressure"] = 5000
MORB["Log fO2 Path"] = "FMQ"
MORB["Increment Temperature"] = -5
MORB["Increment Pressure"] = 0
# %% replicate and add noise
from pyrolite.util.text import slugify
from pyrolite.util.pd import accumulate

reps = 5
df = accumulate([pd.DataFrame(MORB).T] * reps)
df = df.reset_index().drop(columns="index")
compositional = df.pyrochem.oxides
df[compositional] = df[compositional].astype(float).renormalise()
df[compositional] = blur_compositions(df[compositional])

df.Title = df.Title + " " + df.index.map(str)  # differentiate titles
df.Title = df.Title.apply(slugify)
# %% setup an environment for isobaric fractional crystallisation
from pyrolite.ext.alphamelts.env import MELTS_Env

env = MELTS_Env()
env.VERSION = "MELTS"  # crustal processes, < 1GPa/10kbar
env.MODE = "isobaric"
env.DELTAT = -5
env.MINP = 0
env.MAXP = 10000
# %% compositional variation
ax = df.loc[:, ["CaO", "MgO", "Al2O3"]].pyroplot.ternary(alpha=0.2, c="0.5")
# %% save figure
from pyrolite.util.plot import save_figure

save_figure(ax.figure, save_at="../../source/_static", name="melt_blurredmorb")
# %% run the models for each of the inputs
from pyrolite.ext.alphamelts.automation import MeltsBatch

# create a directory to run this experiment in
tempdir = Path("./") / "montecarlo"

batch = MeltsBatch(
    df,
    default_config={
        "Initial Pressure": 5000,
        "Initial Temperature": 1300,
        "Final Temperature": 800,
        "modes": ["isobaric"],
    },
    grid={
        # "Initial Pressure": [3000, 7000],
        "Log fO2 Path": [None, "FMQ"],
        # "modifychem": [None, {"H2O": 0.5}],
    },
    env=env,
    logger=logger,
    fromdir=tempdir,
)

batch.grid  # [{}, {'Log fO2 Path': 'FMQ'}]

batch.run(
    overwrite=True
)  # overwrite=False if you don't want to update existing exp folders

# %% aggregate the results over the same gridded space
from pathlib import Path
from pyrolite.ext.alphamelts.tables import get_experiments_summary
from pyrolite.ext.alphamelts.plottemplates import table_by_phase

tempdir = Path("./") / "montecarlo"

summary = get_experiments_summary(tempdir / "isobar5kbar1300-800C", kelvin=False)
fig = table_by_phase(summary, table="phasemass", plotswide=2, figsize=(10, 8))
# %% save figure
save_figure(fig, save_at="../../source/_static", name="melts_montecarlo")
