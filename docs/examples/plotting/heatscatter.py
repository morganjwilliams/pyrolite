import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot

# %% Minimal Example -------------------------------------------------------------------
# create some example data
from pyrolite.util.synthetic import test_df, random_cov_matrix

df = test_df(
    index_length=1000,
    cov=random_cov_matrix(sigmas=np.random.rand(4) * 2, dim=4, seed=12),
    seed=12,
)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))

df.loc[:, ["SiO2", "MgO"]].pyroplot.scatter(ax=ax[0], c="k", s=10, alpha=0.3)
df.loc[:, ["SiO2", "MgO"]].pyroplot.density(ax=ax[1])
df.loc[:, ["SiO2", "MgO"]].pyroplot.heatscatter(ax=ax[2], s=10, alpha=0.3)
# %% Save Figure
from pyrolite.util.plot import save_figure

for t, a in zip(["Scatter", "Density", "Heatscatter"], ax):
    a.set_title(t)
save_figure(fig, save_at="../../source/_static", name="heatscatter_compare")

# %% Density Ternary -------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(14, 5))

df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.scatter(ax=ax[0], c="k", s=10, alpha=0.1)
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.density(ax=ax[1], bins=100)
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.heatscatter(ax=ax[2], s=5, alpha=0.3)
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="heatscatter_ternary")
