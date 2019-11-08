"""
TAS Classifier
==============

Some simple discrimination methods are implemented,
including the Total Alkali-Silica (TAS) classification.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrolite.util.classification import Geochemistry
from pyrolite.util.synthetic import test_df, random_cov_matrix

########################################################################################
# We'll first generate some synthetic data to play with:
#
mean = np.array([49, 11, 15, 4, 0.5, 4, 1.5])
df = (
    test_df(
        cols=["SiO2", "CaO", "MgO", "FeO", "TiO2", "Na2O", "K2O"],
        mean=mean,
        index_length=100,
        seed=82,
    )
    * mean.sum()
)

df.head(3)
########################################################################################
# We can visualise how this chemistry corresponds to the TAS diagram:
#
from pyrolite.util.classification import Geochemistry

df["TotalAlkali"] = df["Na2O"] + df["K2O"]
cm = Geochemistry.TAS()

fig, ax = plt.subplots(1)

ax.scatter(df["SiO2"], df["TotalAlkali"], c="k", alpha=0.2)
cm.add_to_axes(ax, alpha=0.5, zorder=-1)

########################################################################################
# We can now classify this data according to the fields of the TAS diagram, and
# add this as a column to the dataframe. Similarly, we can extract which rock names
# the TAS fields correspond to:
#
df["TAS"] = cm.classify(df)
df["Rocknames"] = df.TAS.apply(
    lambda x: cm.clsf.fields.get(x, {"names": None})["names"]
)
df["TAS"].unique()
########################################################################################
# We could now take the TAS classes and use them to colorize our points for plotting
# on the TAS diagram, or more likely, on another plot. Here the relationship to the
# TAS diagram is illustrated:
#

colorize = {field: plt.cm.tab10(ix) for ix, field in enumerate(df["TAS"].unique())}

fig, ax = plt.subplots(1)

ax.scatter(
    df["SiO2"], df["TotalAlkali"], c=df["TAS"].apply(lambda x: colorize[x]), alpha=0.7
)
cm.add_to_axes(ax, alpha=0.5, zorder=-1)
