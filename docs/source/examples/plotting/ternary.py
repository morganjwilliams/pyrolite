"""
Ternary Plots
=============
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot

np.random.seed(82)
########################################################################################
# Let's first create some example data:
#
df = pd.DataFrame(data=np.exp(np.random.rand(100, 3)), columns=["SiO2", "MgO", "CaO"])
df.loc[:, ["SiO2", "MgO", "CaO"]].head()
########################################################################################
# Now we can create a simple scatter plot:
#
ax = df.loc[:, ["SiO2", "MgO", "CaO"]].pyroplot.scatter(c="k")
plt.show()
########################################################################################
# If the data represent some continuting, you could also simply plot them as lines:
#
ax = df.loc[:, ["SiO2", "MgO", "CaO"]].pyroplot.plot(color="k", alpha=0.5)
plt.show()
########################################################################################
# The plotting axis can be specified to use exisiting axes:
#
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))

df.loc[:, ["SiO2", "MgO", "CaO"]].sample(20).pyroplot.scatter(ax=ax[0], c="k")
df.loc[:, ["SiO2", "MgO", "CaO"]].sample(20).pyroplot.scatter(ax=ax[1], c="g")

ax = fig.orderedaxes  # creating scatter plots reorders axes, this is the correct order
plt.tight_layout()
########################################################################################
# .. seealso:: `Heatscatter Plots <heatscatter.html>`__,
#              `Density Plots <density.html>`__,
#              `Spider Density Diagrams <conditionaldensity.html>`__
