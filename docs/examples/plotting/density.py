import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot
from pyrolite.plot.density import density
from pyrolite.comp.codata import close

np.random.seed(82)
# %% Minimal Example -------------------------------------------------------------------
# create some example data
oxs = ["SiO2", "CaO", "MgO", "Na2O"]
ys = np.random.rand(1000, len(oxs))
ys[:, 1] += 0.7
ys[:, 2] += 1.0
df = pd.DataFrame(data=close(np.exp(ys)), columns=oxs)
# plot
ax = density(df.loc[:, ["SiO2", "MgO"]].values)
ax.scatter(*df.loc[:, ["SiO2", "MgO"]].values.T, s=10, alpha=0.3, c="k", zorder=2)
# or, alternatively directly from the dataframe:
ax = df.loc[:, ["SiO2", "MgO"]].pyroplot.density()
df.loc[:, ["SiO2", "MgO"]].pyroplot.scatter(ax=ax, s=10, alpha=0.3, c="k", zorder=2)
# %% Save Figure
from pyrolite.util.plot import save_figure

save_figure(ax.figure, save_at="../../source/_static", name="density_minimal")

# %% Colorbar --------------------------------------------------------------------------
ax = df.loc[:, ["SiO2", "MgO"]].pyroplot.density(colorbar=True)
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="density_colorbar")

# %% Specify External Axis -------------------------------------------------------------
# The plotting axis can be specified to use exisiting axes:
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
df.loc[:, ["SiO2", "MgO"]].pyroplot.density(ax=ax[0])
df.loc[:, ["SiO2", "CaO"]].pyroplot.density(ax=ax[1])

plt.tight_layout()
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="density_dual")

# %% Percentiles -----------------------------------------------------------------------
ax = df.loc[:, ["SiO2", "CaO"]].pyroplot.scatter(s=10, alpha=0.3, c="k", zorder=2)
df.loc[:, ["SiO2", "CaO"]].pyroplot.density(ax=ax, contours=[0.95, 0.66, 0.33])
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="density_percentiles")

# %% Logspace --------------------------------------------------------------------------
# some assymetric data -------------
from scipy import stats

xs = stats.norm.rvs(loc=6, scale=3, size=(200, 1))
ys = stats.norm.rvs(loc=20, scale=3, size=(200, 1)) + 5 * xs + 50
data = np.append(xs, ys, axis=1).T
asym_df = pd.DataFrame(np.exp(np.append(xs, ys, axis=1) / 15))
asym_df.columns = ["A", "B"]
grids = ["linxy", "logxy"] * 2 + ["logx", "logy"]
scales = ["linscale"] * 2 + ["logscale"] * 2 + ["semilogx", "semilogy"]
labels = ["{}-{}".format(ls, ps) for (ls, ps) in zip(grids, scales)]
params = list(
    zip(
        [
            (False, False),
            (True, True),
            (False, False),
            (True, True),
            (True, False),
            (False, True),
        ],
        grids,
        scales,
    )
)
# %% ------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(8, 8))
ax = ax.flat

for a, (ls, grid, scale) in zip(ax, params):
    lx, ly = ls
    asym_df.pyroplot.density(ax=a, logx=lx, logy=ly, bins=30, cmap="viridis_r")
    asym_df.pyroplot.density(
        ax=a,
        logx=lx,
        logy=ly,
        contours=[0.95, 0.5],
        bins=30,
        cmap="viridis",
        fontsize=10,
    )
    asym_df.pyroplot.scatter(ax=a, s=10, alpha=0.3, c="k", zorder=2)
    a.set_title("{}-{}".format(grid, scale), fontsize=10)
    if scale in ["logscale", "semilogx"]:
        a.set_xscale("log")
    if scale in ["logscale", "semilogy"]:
        a.set_yscale("log")
plt.tight_layout()
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="density_loggrid")

# %% Modes -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(14, 5))
for a, mode in zip(ax, ["density", "hexbin", "hist2d"]):
    df.loc[:, ["SiO2", "CaO"]].pyroplot.density(ax=a, mode=mode)
    a.set_title("Mode: {}".format(mode))
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="density_modes")

# %% Density Vmin ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, vmin in zip(ax, [0.01, 0.1, 0.4]):
    df.loc[:, ["SiO2", "CaO"]].pyroplot.density(ax=a, bins=30, vmin=vmin, colorbar=True)
plt.tight_layout()
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="density_vmin")

# %% Density Ternary -------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.ternary(ax=ax[0], alpha=0.05, color="k")
for a, mode in zip(ax[1:], ["hist", "density"]):
    df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.density(ax=a, mode=mode, bins=50)
    a.set_title("Mode: {}".format(mode))
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="density_ternary")
