"""
REE Radii Plots
============================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot

# sphinx_gallery_thumbnail_number = 3

########################################################################################
# Here we generate some example data, using the
# :func:`~pyrolite.util.synthetic.example_spider_data` function (based on EMORB,
# here normalised to Primitive Mantle);
#
from pyrolite.util.synthetic import example_spider_data

df = example_spider_data(noise_level=0.1, nobs=20)

########################################################################################
# Where data is specified, the default plot is a line-based spiderplot:
ax = df.pyroplot.REE(color="0.5", figsize=(8, 4))
plt.show()
########################################################################################
# This behaviour can be modified (see spiderplot docs) to provide e.g. filled ranges:
#
df.pyroplot.REE(mode="fill", color="0.5", alpha=0.5, figsize=(8, 4))
plt.show()
########################################################################################
# The plotting axis can be specified to use exisiting axes:
#
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

df.pyroplot.REE(ax=ax[0])
# we can also change the index of the second axes
another_df = example_spider_data(noise_level=0.2, nobs=20)  # some 'nosier' data
another_df.pyroplot.REE(ax=ax[1], color="k", index="radii")

plt.tight_layout()
plt.show()
########################################################################################
# If you're just after a plotting template, you can use
# :func:`~pyrolite.plot.spider.REE_v_radii` to get a formatted axis which can be used
# for subsequent plotting:
#
from pyrolite.plot.spider import REE_v_radii

ax = REE_v_radii(index="radii")  # radii mode will put ionic radii on the x axis
plt.show()
########################################################################################
# .. seealso::
#
#   Examples:
#    `Ionic Radii <ionic_radii.html>`__,
#    `Spider Diagrams <spider.html>`__,
#    `lambdas: Parameterising REE Profiles <lambdas.html>`__
#
#   Functions:
#     :func:`~pyrolite.geochem.ind.get_ionic_radii`,
#     :func:`pyrolite.plot.pyroplot.REE`,
#     :func:`pyrolite.plot.pyroplot.spider`,
#     :func:`~pyrolite.geochem.pyrochem.lambda_lnREE`
