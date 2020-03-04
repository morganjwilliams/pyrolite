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

# sphinx_gallery_thumbnail_number = 3

########################################################################################
# Here we generate some example data, using the
# :func:`~pyrolite.util.synthetic.example_spider_data` function (based on EMORB,
# here normalised to Primitive Mantle);
#
from pyrolite.util.synthetic import example_spider_data

df1 = example_spider_data(noise_level=0.1, nobs=20)
df2 = example_spider_data(noise_level=0.2, nobs=20)
########################################################################################
# Where data is specified, the default plot is a line-based spiderplot:
ax = df1.pyroplot.REE(color="0.5", figsize=(8, 4))
plt.show()
########################################################################################
# This behaviour can be modified (see spiderplot docs) to provide e.g. filled ranges:
#
df1.pyroplot.REE(mode="fill", color="0.5", alpha=0.5, figsize=(8, 4))
plt.show()
########################################################################################
# The plotting axis can be specified to use exisiting axes:
#
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

df1.pyroplot.REE(ax=ax[0])
# we can also change the index of the second axes
df2.pyroplot.REE(ax=ax[1], color="k", index="radii")
plt.tight_layout()
plt.show()

########################################################################################
# If you're just after a plotting template, you can use
# :func:`~pyrolite.plot.spider.REE_v_radii` to get a formatted axis which can be used
# for subsequent plotting:
#
ax = REE_v_radii(index="radii")  # radii mode will put ionic radii on the x axis
plt.show()
########################################################################################
# .. seealso:: `Visualising Orthogonal Polynomials <../lambdas/lambdavis.html>`__,
#              `Dimensional Reduction <../lambdas/lambdadimreduction.html>`__,
#              `Spider Diagrams <spider.html>`__,
