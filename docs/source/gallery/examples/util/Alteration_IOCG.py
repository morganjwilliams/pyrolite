"""
Alteration IOCG (Iron-oxide Copper Gold) Classifier
====================

IOCG alteration discrimination diagram for main IOCG alteration types (Na, Ca-Fe, K-Fe and K)
from Montreuil et al. (2013).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyrolite.util.classification import AIOCG

# sphinx_gallery_thumbnail_number = 2
########################################################################################
# We'll first generate some synthetic magnetite compositions (wt%) to play with:
#
rng = np.random.default_rng(49)
cols = ["K", "Na", "Ca", "Fe", "Mg", "Si", "Mn"]

# cluster medians (molar cations, illustrative only)
centres = {
    "Na":    [0.02, 0.60, 0.20, 0.05, 0.02, 3.0, 0.005],
    "Ca-Fe": [0.05, 0.10, 0.80, 0.60, 0.20, 2.5, 0.05],
    "K-Fe":  [0.60, 0.05, 0.10, 0.50, 0.15, 3.0, 0.02],
    "K":     [0.80, 0.05, 0.05, 0.10, 0.05, 3.2, 0.01],
}

frames = []
for name, mu in centres.items():
    vals = np.exp(np.log(mu) + rng.normal(0, 0.25, size=(50, len(cols))))
    frames.append(pd.DataFrame(vals, columns=cols).assign(true=name))
df = pd.concat(frames, ignore_index=True)

df["(2Ca+5Fe+2Mn)/(2Ca+5Fe+2Mn+Mg+Si) molar"] = (2*df.Ca + 5*df.Fe + 2*df.Mn) / (2*df.Ca + 5*df.Fe + 2*df.Mn + df.Mg + df.Si)
df["K/(K+Na+0.5Ca) molar"] = df.K / (df.K + df.Na + 0.5*df.Ca)
########################################################################################
# We can visualise how this chemistry corresponds to the Alteration IOCG diagram:

cm = AIOCG()

fig, ax = plt.subplots(1)
cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, add_labels=True)
df[["(2Ca+5Fe+2Mn)/(2Ca+5Fe+2Mn+Mg+Si) molar", "K/(K+Na+0.5Ca) molar"]].pyroplot.scatter(ax=ax, c="k", alpha=0.2, axlabels=False)
fig.show()
########################################################################################
# We can now classify this data according to the fields of the diagram, and
# add this as a column to the dataframe:
#
df["AlterationFacies"] = cm.predict(df)
df["AlterationFacies"].sample(10)  # randomly check 10 sample classifications

########################################################################################
# We could now take the classes and use them to colorize our points for plotting
# on the diagram, or more likely, on another plot. Here the relationship to the
# diagram is illustrated, coloring also the populated fields:
#

fig, ax = plt.subplots(1)

cm.add_to_axes(
    ax,
    alpha=0.5,
    linewidth=0.0,
    zorder=-2,
    add_labels=True,
    which_ids=np.unique(df["AlterationFacies"]),
    fill=True,
    facecolor=[0.9, 0.8, 1.0],
)
cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, add_labels=True)
df[["(2Ca+5Fe+2Mn)/(2Ca+5Fe+2Mn+Mg+Si) molar", "K/(K+Na+0.5Ca) molar"]].pyroplot.scatter(
    ax=ax, c=df["AlterationFacies"], alpha=0.7, axlabels=False
)
fig.show()


########################################################################################
# References & Citation
# ~~~~~~~~~~~~~~~~~~~~~
#
#   Montreuil J F, Corriveau L, and Grunsky E C (2013). Compositional data analysis of hydrothermal alteration in
#   IOCG systems, Great Bear magmatic zone, Canada: to each alteration type its own geochemical signature.
#   Geochemistry: Exploration, Environment, Analysis, 13 (4): 229 - 247.
#   `doi: 10.1144/geochem2011-101 <https://doi.org/10.1144/geochem2011-101>`__.

