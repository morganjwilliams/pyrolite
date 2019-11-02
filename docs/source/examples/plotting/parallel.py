"""
Parallel Coordinate Plots
============================

Parallel coordinate plots are one way to visualise data relationships and clusters in
higher dimensional data. pyrolite now includes an implementation of this which allows
a handy quick exploratory visualisation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
import pyrolite.plot
import pyrolite.data.Aitchison
# sphinx_gallery_thumbnail_number = 3

########################################################################################
# To start, let's load up an example dataset from Aitchison
#
df = pyrolite.data.Aitchison.load_coxite()
comp = [
    i for i in df.columns if i not in ["Depth", "Porosity"]
]  # compositional data variables
########################################################################################
ax = df.pyroplot.parallel()
plt.show()
########################################################################################
# By rescaling this using the mean and standard deviation, we can account for scale
# differences between variables:
#
ax = df.pyroplot.parallel(rescale=True)
plt.show()
########################################################################################
# We can also use a centred-log transform for compositional data to reduce the effects
# of spurious correlation:
#
from pyrolite.util.skl.transform import CLRTransform

cmap = "inferno"
compdata = df.copy()
compdata[comp] = CLRTransform().transform(compdata[comp])
ax = compdata.loc[:, comp].pyroplot.parallel(color_by=compdata.Depth.values, cmap=cmap)

# we can add a meaningful colorbar to indicate one variable also, here Depth
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array(df.Depth)
plt.colorbar(sm)
plt.show()
########################################################################################
ax = compdata.loc[:, comp].pyroplot.parallel(
    rescale=True, color_by=compdata.Depth.values, cmap=cmap
)
plt.colorbar(sm)
plt.show()
