import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import spiderplot
from pyrolite.geochem.ind import common_elements

np.random.seed(82)
# %% Minimal Example -------------------------------------------------------------------
# create some example data
els = common_elements(cutoff=47)
ys = np.random.rand(3, len(els))
ys = np.exp(ys)
df = pd.DataFrame(data=ys, columns=els)

ax = spiderplot(df.loc[0, :], color="k")
# or, alternatively directly from the dataframe:
ax = df.loc[0, :].spiderplot(color="k")
ax.set_ylabel('Abundance')
# %% Save Figure
from pyrolite.util.plot import save_figure
save_figure(ax.figure, save_at="../../source/_static", name="spiderplot_minimal")

# %% Fill Plot -------------------------------------------------------------------------
# This behaviour can be modified (see spiderplot docs) to provide filled ranges:
ax = spiderplot(df, fill=True, plot=False, color="k", alpha=0.5)
# or, alternatively directly from the dataframe:
ax = df.spiderplot(fill=True, plot=False, color="k", alpha=0.5)
ax.set_ylabel('Abundance')
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="spiderplot_fill")

# %% Specify External Axis ------------------------------------------------------------
# The plotting axis can be specified to use exisiting axes:
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6))
ax[0].set_ylabel('Abundance')

df.spiderplot(ax=ax[0], color="k")
df.spiderplot(ax=ax[1], fill=True, plot=False, color="k", alpha=0.5)

plt.tight_layout()
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="spiderplot_dual")

# %% Normalised Data -------------------------------------------------------------------
# Spiderplots are most commonly used to disply normalised abundances. This is easily
# accomplished using pyrolite.norm:
from pyrolite.norm import ReferenceCompositions
rc = ReferenceCompositions()['Chondrite_PON']
normdf = rc.normalize(df)

ax = spiderplot(normdf, color="k")
# or, alternatively directly from the dataframe:
ax = normdf.spiderplot(color="k")

ax.set_ylabel('Abundance / Chondrite')
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="spiderplot_norm")
