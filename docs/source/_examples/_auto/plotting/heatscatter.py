"""
Heatscatter Plots
==================================

While :func:`~pyrolite.plot.pyroplot.density` plots are useful summary visualizations
for large datasets, scatterplots are more precise and retain all spatial information
(although they can get crowded).

A scatter plot where individual points are coloured by data density in some respects
represents the best of both worlds. A version inspired by similar existing
visualisations is implemented with :func:`~pyrolite.plot.pyroplot.heatscatter`.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot

np.random.seed(12)
########################################################################################
# create some example data
from pyrolite.util.synthetic import test_df, random_cov_matrix

df = test_df(
    index_length=1000,
    cov=random_cov_matrix(sigmas=np.random.rand(4) * 2, dim=4, seed=12),
    seed=12,
)

########################################################################################
# We can compare a minimal :func:`~pyrolite.plot.pyroplot.heatscatter` plot to other
# visualisations for the same data:
#
from pyrolite.util.plot import share_axes

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
ax = ax.flat
share_axes(ax[:3], which="xy")
share_axes(ax[3:6], which="xy")
share_axes(ax[6:], which="xy")

# linear-scaled comparison
df.loc[:, ["SiO2", "MgO"]].pyroplot.scatter(ax=ax[0], c="k", s=10, alpha=0.3)
df.loc[:, ["SiO2", "MgO"]].pyroplot.density(ax=ax[1])
df.loc[:, ["SiO2", "MgO"]].pyroplot.heatscatter(ax=ax[2], s=10, alpha=0.3)

# log-log plots
df.loc[:, ["SiO2", "MgO"]].pyroplot.scatter(ax=ax[3], c="k", s=10, alpha=0.3)
df.loc[:, ["SiO2", "MgO"]].pyroplot.density(ax=ax[4], logx=True, logy=True)
df.loc[:, ["SiO2", "MgO"]].pyroplot.heatscatter(
    ax=ax[5], s=10, alpha=0.3, logx=True, logy=True
)
# ternary plots
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.scatter(ax=ax[6], c="k", s=10, alpha=0.1)
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.density(ax=ax[7], bins=100)
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.heatscatter(
    ax=ax[8], s=10, alpha=0.3, renorm=True
)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

titles = ["Scatter", "Density", "Heatscatter"]
for t, a in zip(titles + [i + " (log-log)" for i in titles], ax):
    a.set_title(t)
########################################################################################
# .. seealso:: `Ternary Plots <ternary.html>`__,
#              `Density Plots <density.html>`__,
#              `Spider Density Diagrams <conditionaldensity.html>`__