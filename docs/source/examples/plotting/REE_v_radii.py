"""
REE Radii Plots
============================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot
from pyrolite.plot.spider import REE_v_radii
from pyrolite.geochem.ind import REE, get_ionic_radii
# sphinx_gallery_thumbnail_number = 4

########################################################################################
# Where data is not specified, it will return a formatted axis which can be used for
# subsequent plotting:
ax = REE_v_radii(index='radii') # radii mode will put ionic radii on the x axis

# create some example data
ree = REE()
xs = get_ionic_radii(ree, coordination=8, charge=3)
ys = np.linspace(1, 20, len(xs))
ax.plot(xs, ys, marker='D', color='k')
plt.show()
########################################################################################
# Here we generate some example data:
#
no_analyses = 10

data_ree = REE(dropPm=True)
data_radii = np.array(get_ionic_radii(data_ree, coordination=8, charge=3))
data_radii = np.tile(data_radii, (1, no_analyses)).reshape(
    no_analyses, data_radii.shape[0]
)

dataframes = []

for i in range(2):
    noise = np.random.randn(*data_radii.shape) * 0.1
    constant = -0.1
    lin = np.tile(np.linspace(3.0, 0.0, data_radii.shape[1]), (no_analyses, 1))
    lin = (lin.T * (1.1 + i/2 * np.random.rand(data_radii.shape[0]))).T
    quad = -1.2 * (data_radii - 1.11) ** 2.0

    lnY = noise + constant + lin + quad

    for ix, el in enumerate(data_ree):
        if el in ["Ce", "Eu"]:
            lnY[:, ix] += np.random.rand(no_analyses) * 0.6

    Y = np.exp(lnY)

    df = pd.DataFrame(Y, columns=data_ree)
    dataframes.append(df)

df1 = dataframes[0]
df2 = dataframes[1]
########################################################################################
# Where data is specified, the default plot is a line-based spiderplot:
ax = REE_v_radii(df1.values, ree=data_ree)

# or, alternatively directly from the dataframe:
ax = df1.pyroplot.REE()
plt.show()
########################################################################################
# This behaviour can be modified (see spiderplot docs) to provide filled ranges:
#
ax = REE_v_radii(df1.values, ree=data_ree, mode='fill')
# or, alternatively directly from the dataframe:
ax = df1.pyroplot.REE(mode='fill')
plt.show()
########################################################################################
# The plotting axis can be specified to use exisiting axes:
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

df1.pyroplot.REE(ax=ax[0])
# we can also change the index of the second figure
ax1 = df2.pyroplot.REE(ax=ax[1], color='k', index='radii')
plt.tight_layout()
plt.show()
########################################################################################
# .. seealso:: `Visualising Orthogonal Polynomials <../lambdas/lambdavis.html>`__,
#              `Dimensional Reduction <../lambdas/lambdadimreduction.html>`__,
#              `Spider Diagrams <spider.html>`__,
