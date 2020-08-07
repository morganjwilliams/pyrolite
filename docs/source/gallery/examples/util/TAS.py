"""
TAS Classifier
==============

Some simple discrimination methods are implemented,
including the Total Alkali-Silica (TAS) classification.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrolite.util.classification import TAS
from pyrolite.util.synthetic import normal_frame, random_cov_matrix

# sphinx_gallery_thumbnail_number = 2
########################################################################################
# We'll first generate some synthetic data to play with:
#
df = (
    normal_frame(
        columns=["SiO2", "Na2O", "K2O", "Al2O3"],
        mean=[0.5, 0.04, 0.05, 0.4],
        size=100,
        seed=49,
    )
    * 100
)

df.head(3)
########################################################################################
# We can visualise how this chemistry corresponds to the TAS diagram:
#
import pyrolite.plot

df["Na2O + K2O"] = df["Na2O"] + df["K2O"]
cm = TAS()

fig, ax = plt.subplots(1)
cm.add_to_axes(
    ax, alpha=0.5, linewidth=0.5, zorder=-1, labels="ID",
)
df[["SiO2", "Na2O + K2O"]].pyroplot.scatter(ax=ax, c="k", alpha=0.2)
plt.show()
########################################################################################
# We can now classify this data according to the fields of the TAS diagram, and
# add this as a column to the dataframe. Similarly, we can extract which rock names
# the TAS fields correspond to:
#
df["TAS"] = cm.predict(df)
df["Rocknames"] = df.TAS.apply(lambda x: cm.fields.get(x, {"name": None})["name"])
df["Rocknames"].sample(10)  # randomly check 10 sample rocknames
########################################################################################
# We could now take the TAS classes and use them to colorize our points for plotting
# on the TAS diagram, or more likely, on another plot. Here the relationship to the
# TAS diagram is illustrated:
#

fig, ax = plt.subplots(1)

cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, labels="ID")
df[["SiO2", "Na2O + K2O"]].pyroplot.scatter(ax=ax, c=df["TAS"], alpha=0.7)
