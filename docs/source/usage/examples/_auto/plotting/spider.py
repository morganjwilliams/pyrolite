"""
Spiderplots & Density Spiderplots
==================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 4

########################################################################################
# Here we'll set up an example which uses EMORB as a starting point:
#
from pyrolite.geochem.norm import get_reference_composition

ref = get_reference_composition("EMORB_SM89")  # emorb composition as a starting point
ref.set_units("ppm")
df = ref.comp.pyrochem.compositional
########################################################################################
# Basic spider plots are straightforward to produce:
import pyrolite.plot

ax = df.pyroplot.spider(color="k")
plt.show()
########################################################################################
# Typically we'll normalise trace element compositions to a reference composition
# to be able to link the diagram to 'relative enrichement' occuring during geological
# processes:
#
normdf = df.pyrochem.normalize_to("PM_PON", units="ppm")
ax = normdf.pyroplot.spider(color="k", unity_line=True)
plt.show()
########################################################################################
# The spiderplot can be extended to provide visualisations of ranges and density via the
# various modes. First let's take this composition and add some noise in log-space to
# generate multiple compositions about this mean (i.e. a compositional distribution):
#
start = normdf.applymap(np.log)
nindex, nobs = normdf.columns.size, 120

noise_level = 0.5  # sigma for noise
x = np.arange(nindex)
y = np.tile(start.values, nobs).reshape(nobs, nindex)
y += np.random.normal(0, noise_level / 2.0, size=(nobs, nindex))  # noise
y += np.random.normal(0, noise_level, size=(1, nobs)).T  # random pattern offset

distdf = pd.DataFrame(y, columns=normdf.columns)
distdf["Eu"] += 1.0  # significant offset for Eu anomaly
distdf = distdf.applymap(np.exp)
########################################################################################
# We could now plot the range of compositions as a filled range:
#
ax = distdf.pyroplot.spider(mode="fill", color="green", alpha=0.5, unity_line=True)
plt.show()
########################################################################################
# Alternatively, we can plot a conditional density spider plot:
#
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6))
_ = distdf.pyroplot.spider(ax=ax[0], color="k", alpha=0.05, unity_line=True)
_ = distdf.pyroplot.spider(
    ax=ax[1],
    mode="binkde",
    cmap="viridis",
    vmin=0.05,  # minimum percentile,
    resolution=10,
    unity_line=True
)
########################################################################################
# We can now assemble a more complete comparison of some of the conditional density
# modes for spider plots:
#
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
down, across = len(modes), 1
fig, ax = plt.subplots(
    down, across, sharey=True, sharex=True, figsize=(across * 8, 2 * down)
)

for a, (m, name, args, kwargs) in zip(ax, modes):
    a.annotate(  # label the axes rows
        "Mode: {}".format(name),
        xy=(0.1, 1.05),
        xycoords=a.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )
ax = ax.flat
for mix, (m, name, args, kwargs) in enumerate(modes):
    distdf.pyroplot.spider(
        mode=m,
        ax=ax[mix],
        cmap="viridis",
        vmin=0.05,  # minimum percentile
        fontsize=8,
        unity_line=True,
        *args,
        **kwargs
    )

plt.tight_layout()
########################################################################################
# .. seealso:: `Heatscatter Plots <heatscatter.html>`__,
#              `Density Diagrams <density.html>`__
