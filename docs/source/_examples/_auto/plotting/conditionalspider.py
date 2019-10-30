"""
Conditional Density Spiderplots
==================================

The spiderplot can be extended to provide visualisations of ranges and density via the
various modes:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot
from pyrolite.plot.spider import spider
from pyrolite.geochem import REE, get_reference_composition
import logging

rn = get_reference_composition("EMORB_SM89")  # emorb composition as a starting point
rn.set_units("ppm")
data = rn.comp[REE(dropPm=True)]
########################################################################################
nindex, nobs = data.size, 120
ss = [0.1, 0.2, 0.5]  # sigmas for noise

modes = [
    ("plot", "plot", [], dict(color="k", alpha=0.01)),
    ("fill", "fill", [], dict(color="k", alpha=0.5)),
    ("binkde", "binkde", [], dict(resolution=10)),
    (
        "binkde",
        "binkde contours specified",
        [],
        dict(contours=[0.95], resolution=10),  # 95th percentile contour
    ),
    ("histogram", "histogram", [], dict(resolution=5, ybins=30)),
]
########################################################################################
fig, ax = plt.subplots(
    len(modes), len(ss), sharey=True, figsize=(len(ss) * 3, 2 * len(modes))
)
ax[0, 0].set_ylim((0.1, 100))

for a, (m, name, args, kwargs) in zip(ax, modes):
    a[0].annotate(  # label the axes rows
        "Mode: {}".format(name),
        xy=(0.1, 1.05),
        xycoords=a[0].transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )

start = data.pyrochem.normalize_to("PM_PON", units="ppm").applymap(np.log)
for ix, s in enumerate(ss):
    x = np.arange(nindex)
    y = np.tile(start.values, nobs).reshape(nobs, nindex)
    y += np.random.normal(0, s / 2.0, size=(nobs, nindex))  # noise
    y += np.random.normal(0, s, size=(1, nobs)).T  # random pattern offset

    df = pd.DataFrame(y, columns=data.columns)
    df["Eu"] += 1.0  # significant offset
    df = df.applymap(np.exp)
    for mix, (m, name, args, kwargs) in enumerate(modes):
        df.pyroplot.spider(
            mode=m,
            ax=ax[mix, ix],
            cmap="viridis",
            vmin=0.05,  # minimum percentile
            *args,
            **kwargs
        )

plt.tight_layout()
########################################################################################
# .. seealso:: `Heatscatter Plots <heatscatter.html>`__,
#              `Spider Plots <spider.html>`__,
#              `Density Diagrams <density.html>`__
