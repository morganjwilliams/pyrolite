"""
Spider Plots
==============
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot
from pyrolite.plot.spider import spider
from pyrolite.geochem.ind import common_elements

np.random.seed(82)
########################################################################################
els = common_elements(cutoff=47)[10:]
ys = np.random.rand(3, len(els))
ys = np.exp(ys)
df = pd.DataFrame(data=ys, columns=els)

ax = df.loc[0, :].pyroplot.spider(color="k")
ax.set_ylabel("Abundance")
plt.show()
########################################################################################
# This behaviour can be modified (see spider docs) to provide filled ranges:
ax = df.pyroplot.spider(mode="fill", color="k", alpha=0.5)
ax.set_ylabel("Abundance")
plt.show()
########################################################################################
# The plotting axis can be specified to use exisiting axes:
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6))
ax[0].set_ylabel("Abundance")

df.pyroplot.spider(ax=ax[0], color="k")
df.pyroplot.spider(ax=ax[1], mode="fill", color="k", alpha=0.5)

plt.tight_layout()
########################################################################################
# Spidergrams are most commonly used to disply normalised abundances. This is easily
# accomplished using :mod:`pyrolite.geochem.norm`:

normdf = df.pyrochem.normalize_to("Chondrite_PON", units="ppm")

ax = spider(normdf.values, color="k")
# or, alternatively directly from the dataframe:
ax = normdf.pyroplot.spider(color="k")

ax.set_ylabel("Abundance / Chondrite")
plt.show()
########################################################################################
# .. seealso:: `Spider Density Diagrams <conditionaldensity.html>`__,
#              `Normalisation <../geochem/normalization.html>`__,
#              `REE Radii Plot <REE_v_radii.html>`__,
#              `REE Dimensional Reduction <../lambdas/lambdadimreduction.html>`__
