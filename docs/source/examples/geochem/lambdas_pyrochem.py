"""
lambdas: pyrochem
===================
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot
from pyrolite.geochem.ind import REE, get_ionic_radii
from pyrolite.geochem.transform import lambda_lnREE
from pyrolite.plot.spider import REE_v_radii
from pyrolite.util.math import lambdas, lambda_poly_func, OP_constants
# sphinx_gallery_thumbnail_number = 1

np.random.seed(82)
########################################################################################
# First we generate some example REE data, and plot this on a
# :func:`pyrolite.plot.REE_v_radii`:
#
no_analyses = 1000
data_ree = REE(dropPm=True)
data_radii = np.array(get_ionic_radii(data_ree, charge=3, coordination=8))
data_radii = np.tile(data_radii, (1, no_analyses)).reshape(
    no_analyses, data_radii.shape[0]
)

noise = np.random.randn(*data_radii.shape) * 0.1
constant = -0.1
lin = np.tile(np.linspace(3.0, 0.0, data_radii.shape[1]), (no_analyses, 1))
lin = (lin.T * (1.1 + 0.4 * np.random.rand(data_radii.shape[0]))).T
quad = -1.2 * (data_radii - 1.11) ** 2.0

lnY = noise + constant + lin + quad

for ix, el in enumerate(data_ree):
    if el in ["Ce", "Eu"]:
        lnY[:, ix] += np.random.rand(no_analyses) * 0.6
data_radii
df = pd.DataFrame(np.exp(lnY), columns=data_ree)
ax = df.pyroplot.REE(marker="D", alpha=0.01, c="0.5", markerfacecolor="k")
########################################################################################
# The reduction to lambdas using the pandas interface is much simpler than using the
# numpy-based utility functions (see :func:`pyrolite.util.math.lambdas`,
# `Dimensional Reduction <lambdadimreduction.html>`__):
#
ls = df.pyrochem.lambda_lnREE(
    exclude=["Ce", "Eu", "Pm"], degree=4, norm_to="Chondrite_PON"
)
########################################################################################
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax_labels = [chr(955) + "$_{}$".format(str(d)) for d in range(4)]
columns = [chr(955) + str(d) for d in range(4)]

for ix, a in enumerate(ax):
    a.scatter(ls[columns[ix]], ls[columns[ix + 1]], alpha=0.1, c="k")
    a.set_xlabel(ax_labels[ix])
    a.set_ylabel(ax_labels[ix + 1])

plt.tight_layout()
fig.suptitle("Lambdas for Dimensional Reduction", y=1.05)
########################################################################################
# For more on using orthogonal polynomials to describe geochemical pattern data, see:
# O’Neill, H.S.C., 2016. The Smoothness and Shapes of Chondrite-normalized Rare Earth
# Element Patterns in Basalts. J Petrology 57, 1463–1508.
# `doi: 10.1093/petrology/egw047 <https://doi.org/10.1093/petrology/egw047>`__.
#
# .. seealso::
#
#   Examples:
#     `Visualising Orthogonal Polynomials <lambdavis.html>`__,
#     `Dimensional Reduction <lambdadimreduction.html>`__,
#     `REE Radii Plot <../plotting/REE_v_radii.html>`__
