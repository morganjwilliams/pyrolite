import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot

np.random.seed(12)
# %% Minimal Example -------------------------------------------------------------------
# create some example data
from pyrolite.util.synthetic import test_df, random_cov_matrix

df = test_df(
    index_length=1000,
    cov=random_cov_matrix(sigmas=np.random.rand(4) * 2, dim=4, seed=12),
    seed=12,
)

# %%
from pyrolite.util.plot import share_axes

fig, ax = plt.subplots(2, 3, figsize=(10, 6))
ax = ax.flat
share_axes(ax[:3], which="xy")
share_axes(ax[3:], which="xy")

df.loc[:, ["SiO2", "MgO"]].pyroplot.scatter(ax=ax[0], c="k", s=10, alpha=0.3)
df.loc[:, ["SiO2", "MgO"]].pyroplot.density(ax=ax[1])
df.loc[:, ["SiO2", "MgO"]].pyroplot.heatscatter(ax=ax[2], s=10, alpha=0.3)

df.loc[:, ["SiO2", "MgO"]].pyroplot.scatter(ax=ax[3], c="k", s=10, alpha=0.3)
df.loc[:, ["SiO2", "MgO"]].pyroplot.density(ax=ax[4], logx=True, logy=True)
df.loc[:, ["SiO2", "MgO"]].pyroplot.heatscatter(
    ax=ax[5], s=10, alpha=0.3, logx=True, logy=True
)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
# %% Save Figure
from pyrolite.util.plot import save_figure
titles = ["Scatter", "Density", "Heatscatter"]
for t, a in zip(titles+[i+" (log-log)" for i in titles], ax):
    a.set_title(t)

save_figure(fig, save_at="../../source/_static", name="heatscatter_compare")

# %% Density Ternary -------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(14, 5))

df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.scatter(ax=ax[0], c="k", s=10, alpha=0.1)
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.density(ax=ax[1], bins=100)
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.heatscatter(
    ax=ax[2], s=10, alpha=0.3, renorm=True
)
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="heatscatter_ternary")
