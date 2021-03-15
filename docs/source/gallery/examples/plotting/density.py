"""
Density and Contour Plots
==================================

While individual point data are useful, we commonly want to understand the
the distribution of our data within a particular subspace, and compare that
to a reference or other dataset. Pyrolite includes a few functions for
visualising data density, most based on Gaussian kernel density estimation
and evaluation over a grid. The below examples highlight some of the currently
implemented features.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot
from pyrolite.plot.density import density
from pyrolite.comp.codata import close

# sphinx_gallery_thumbnail_number = 6

np.random.seed(82)
########################################################################################
# First we create some example data :
#
oxs = ["SiO2", "CaO", "MgO", "Na2O"]
ys = np.random.rand(1000, len(oxs))
ys[:, 1] += 0.7
ys[:, 2] += 1.0
df = pd.DataFrame(data=close(np.exp(ys)), columns=oxs)
########################################################################################
# A minimal density plot can be constructed as follows:
#
ax = df.loc[:, ["SiO2", "MgO"]].pyroplot.density()
df.loc[:, ["SiO2", "MgO"]].pyroplot.scatter(ax=ax, s=10, alpha=0.3, c="k", zorder=2)
plt.show()
########################################################################################
# A colorbar linked to the KDE estimate colormap can be added using the `colorbar`
# boolean switch:
#
ax = df.loc[:, ["SiO2", "MgO"]].pyroplot.density(colorbar=True)
plt.show()
########################################################################################
# `density` by default will create a new axis, but can also be plotted over an
# existing axis for more control:
#
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))

df.loc[:, ["SiO2", "MgO"]].pyroplot.density(ax=ax[0])
df.loc[:, ["SiO2", "CaO"]].pyroplot.density(ax=ax[1])

plt.tight_layout()
plt.show()
########################################################################################
# Contours are also easily created, which by default are percentile values:
#
ax = df.loc[:, ["SiO2", "CaO"]].pyroplot.scatter(s=10, alpha=0.3, c="k", zorder=2)
df.loc[:, ["SiO2", "CaO"]].pyroplot.density(ax=ax, contours=[0.95, 0.66, 0.33])
plt.show()
########################################################################################
# Geochemical data is commonly log-normally distributed and is best analysed
# and visualised after log-transformation. The density estimation can be conducted
# over logspaced grids (individually for x and y axes using `logx` and `logy` boolean
# switches). Notably, this makes both the KDE image and contours behave more naturally:
#

# some assymetric data
from scipy import stats

xs = stats.norm.rvs(loc=6, scale=3, size=(200, 1))
ys = stats.norm.rvs(loc=20, scale=3, size=(200, 1)) + 5 * xs + 50
data = np.append(xs, ys, axis=1).T
asym_df = pd.DataFrame(np.exp(np.append(xs, ys, axis=1) / 25.0))
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
########################################################################################
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
plt.show()
#######################################################################################
plt.close("all")  # let's save some memory..
########################################################################################
# There are two other implemented modes beyond the default `density`: `hist2d` and
# `hexbin`, which parallel their equivalents in matplotlib.
# Contouring is not enabled for these histogram methods.
#
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(14, 5))
for a, mode in zip(ax, ["density", "hexbin", "hist2d"]):
    df.loc[:, ["SiO2", "CaO"]].pyroplot.density(ax=a, mode=mode)
    a.set_title("Mode: {}".format(mode))
plt.show()
########################################################################################
# For the ``density`` mode, a ``vmin`` parameter is used to choose the lower
# threshold, and by default is the 99th percentile (``vmin=0.01``), but can be
# adjusted. This is useful where there are a number of outliers, or where you wish to
# reduce the overall complexity/colour intensity of a figure (also good for printing!).
#
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, vmin in zip(ax, [0.01, 0.1, 0.4]):
    df.loc[:, ["SiO2", "CaO"]].pyroplot.density(ax=a, bins=30, vmin=vmin, colorbar=True)
plt.tight_layout()
plt.show()
#######################################################################################
plt.close("all")  # let's save some memory..
########################################################################################
# Density plots can also be used for ternary diagrams, where more than two components
# are specified:
#
fig, ax = plt.subplots(
    1,
    3,
    sharex=True,
    sharey=True,
    figsize=(15, 5),
    subplot_kw=dict(projection="ternary"),
)
df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.scatter(ax=ax[0], alpha=0.05, c="k")
for a, mode in zip(ax[1:], ["hist", "density"]):
    df.loc[:, ["SiO2", "CaO", "MgO"]].pyroplot.density(ax=a, mode=mode)
    a.set_title("Mode: {}".format(mode), y=1.2)

plt.tight_layout()
plt.show()
########################################################################################
# .. note:: Using alpha with the ``density`` mode induces a known and old matplotlib bug,
#           where the edges of bins within a ``pcolormesh`` image (used for plotting the
#           KDE estimate) are over-emphasized, giving a gridded look.
#
# .. seealso:: `Heatscatter Plots <heatscatter.html>`__,
#              `Ternary Plots <ternary.html>`__,
#              `Spider Density Diagrams <spider.html>`__
