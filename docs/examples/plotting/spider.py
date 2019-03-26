import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot.spider import spider
from pyrolite.geochem.ind import common_elements

np.random.seed(82)
# %% Minimal Example -------------------------------------------------------------------
# create some example data
els = common_elements(cutoff=47)
ys = np.random.rand(3, len(els))
ys = np.exp(ys)
df = pd.DataFrame(data=ys, columns=els)

ax = spider(df.loc[0, :].values, color="k")
# or, alternatively directly from the dataframe:
ax = df.loc[0, :].pyroplot.spider(color="k")
ax.set_ylabel('Abundance')
# %% Save Figure
from pyrolite.util.plot import save_figure
save_figure(ax.figure, save_at="../../source/_static", name="spider_minimal")

# %% Fill Plot -------------------------------------------------------------------------
# This behaviour can be modified (see spider docs) to provide filled ranges:
ax = spider(df.values, mode='fill', color="k", alpha=0.5)
# or, alternatively directly from the dataframe:
ax = df.pyroplot.spider(mode='fill', color="k", alpha=0.5)
ax.set_ylabel('Abundance')
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="spider_fill")

# %% Specify External Axis ------------------------------------------------------------
# The plotting axis can be specified to use exisiting axes:
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6))
ax[0].set_ylabel('Abundance')

df.pyroplot.spider(ax=ax[0], color="k")
df.pyroplot.spider(ax=ax[1], fill=True, plot=False, color="k", alpha=0.5)

plt.tight_layout()
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="spider_dual")

# %% Normalised Data -------------------------------------------------------------------
# spiders are most commonly used to disply normalised abundances. This is easily
# accomplished using pyrolite.norm:
from pyrolite.geochem.norm import ReferenceCompositions
rc = ReferenceCompositions()['Chondrite_PON']
normdf = rc.normalize(df)

ax = spider(normdf.values, color="k")
# or, alternatively directly from the dataframe:
ax = normdf.pyroplot.spider(color="k")

ax.set_ylabel('Abundance / Chondrite')
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="spider_norm")
