"""
Magnetite Classifier
====================

Magnetite trace element discrimination diagrams for mineral deposit types
(Dupuis and Beaudoin, 2011) are implemented as log-log polygon classifiers.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyrolite.util.classification import Magnetite
from pyrolite.util.synthetic import normal_frame

# sphinx_gallery_thumbnail_number = 2
########################################################################################
# We'll first generate some synthetic magnetite compositions (wt%) to play with:
#
df = (
    normal_frame(
        columns=["Fe", "Ti", "V", "Ca", "Al", "Mn", "Ni", "Cr"],
        mean=[0.99, 0.002, 0.001, 0.0005, 0.002, 0.001, 0.0003, 0.0005],
        cov=np.eye(7) * 0.8,
        size=100,
        seed=49,
    )
    * 100
)

df.head(3)
########################################################################################
# We can visualise how this chemistry corresponds to the Ca+Al+Mn vs Ti+V diagram:
#

df["Ti+V"] = df["Ti"] + df["V"]
df["Ca+Al+Mn"] = df["Ca"] + df["Al"] + df["Mn"]
df["Ni/(Cr+Mn)"] = df["Ni"] / (df["Cr"] + df["Mn"])
cm = Magnetite()

fig, ax = plt.subplots(1)
cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, add_labels=True)
df[["Ti+V", "Ca+Al+Mn"]].pyroplot.scatter(ax=ax, c="k", alpha=0.2, axlabels=False)
plt.show()
########################################################################################
# We can now classify this data according to the fields of the diagram, and
# add this as a column to the dataframe:
#
df["Magnetite"] = cm.predict(df)
df["Magnetite"].sample(10)  # randomly check 10 sample classifications
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
    add_labels=False,
    which_ids=np.unique(df["Magnetite"]),
    fill=True,
    facecolor=[0.9, 0.8, 1.0],
)
cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, add_labels=True)
df[["Ti+V", "Ca+Al+Mn"]].pyroplot.scatter(
    ax=ax, c=df["Magnetite"], alpha=0.7, axlabels=False
)

########################################################################################
# Variations of the Diagram
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To use the other diagram, pass the relevant keyword :code:`which_model`.
# Currently, the Ca+Al+Mn vs Ti+V (:code:`"TiV_CaAlMn"`, default) and
# Ni/(Cr+Mn) vs Ti+V (:code:`"TiV_NiCrMn"`) diagrams are available. Each of these
# can be used as a classifier model as for the default above.
#
models = {"TiV_CaAlMn": "Ca+Al+Mn", "TiV_NiCrMn": "Ni/(Cr+Mn)"}
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, (model, y) in zip(ax, models.items()):
    a.set_title(model)
    cm = Magnetite(which_model=model)
    cm.add_to_axes(a, alpha=0.5, linewidth=0.5, add_labels=True)
    df[["Ti+V", y]].pyroplot.scatter(ax=a, c="k", alpha=0.2, axlabels=False)
plt.show()
########################################################################################
# References & Citation
# ~~~~~~~~~~~~~~~~~~~~~
#
#   Dupuis, C., Beaudoin, G. (2011). Discriminant diagrams for iron oxide trace element fingerprinting of
#   mineral deposit types. Mineralium Deposita 46, 319-335.
#   `doi: 10.1007/s00126-011-0334-y <https://doi.org/10.1007/s00126-011-0334-y>`__.
#
#   Nadoll, P. (2011). Geochemistry of magnetite from hydrothermal ore deposits and host rocks –
#   Case studies from the Proterozoic Belt Supergroup, Cu-Mo-porphyry + skarn and Climax-Mo deposits
#   in the western United States.
#   `doi: hdl.handle.net/2292/7512 <https://hdl.handle.net/2292/7512>`__.
#
#   Nadoll, P., Angerer, T., Mauk, J. L., French, D., & Walshe, J. (2014). The chemistry of hydrothermal
#   magnetite: A review. Ore Geology Reviews, 61, 1–32.
#   `doi: 10.1016/j.oregeorev.2013.12.013 <https://doi.org/10.1016/j.oregeorev.2013.12.013>`__.