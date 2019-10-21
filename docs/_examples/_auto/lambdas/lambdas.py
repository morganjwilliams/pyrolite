"""
Dimensional Reduction
======================

Orthogonal polynomial decomposition can be used for dimensional reduction of smooth
function over an independent variable, producing an array of independent values
representing the relative weights for each order of component polynomial.

In geochemistry, the most applicable use case is for reduction Rare Earth Element (REE)
profiles. The REE are a collection of elements with broadly similar physicochemical
properties (the lanthanides), which vary with ionic radii. Given their similar behaviour
and typically smooth function of normalised abundance vs. ionic radii, the REE profiles
and their shapes can be effectively parameterised and dimensionally reduced (14 elements
summarised by 3-4 shape parameters).

Here we generate some example data, reduce these to lambda values, and plot the
resulting dimensionally reduced data.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pyrolite.geochem.ind import REE, get_ionic_radii
from pyrolite.plot.spider import REE_v_radii
from pyrolite.util.math import lambdas, lambda_poly_func, OP_constants
# sphinx_gallery_thumbnail_number = 2

np.random.seed(82)
########################################################################################
# First we'll generate some example data:
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

Y = np.exp(lnY)

ax = REE_v_radii(
    Y,
    ree=data_ree,
    marker="D",
    alpha=0.01,
    color="0.5",
    markerfacecolor="k",
    index="elements",
)
########################################################################################
# From this data we can calculate and plot the lambda values:
#
lambda_degree = 4

exclude = ["Ce", "Eu", "Pm"]
if exclude:
    subset_Y = Y[:, [i not in exclude for i in data_ree]]
    subset_ree = [i for i in REE() if not i in exclude]
    subset_radii = np.array(get_ionic_radii(subset_ree, charge=3, coordination=8))

params = OP_constants(subset_radii, degree=lambda_degree)

ls = np.apply_along_axis(
    lambda x: lambdas(x, subset_radii, params=params, degree=4), 1, np.log(subset_Y)
)

#--------------------------------------------------------------------------------------

fig, ax = plt.subplots(1, lambda_degree - 1, figsize=(9, 3))
ax_labels = [chr(955) + "$_{}$".format(str(d)) for d in range(lambda_degree)]

for ix in range(lambda_degree - 1):
    ax[ix].scatter(ls[:, ix], ls[:, ix + 1], alpha=0.1, c="k")
    ax[ix].set_xlabel(ax_labels[ix])
    ax[ix].set_ylabel(ax_labels[ix + 1])

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
#     `Pandas Lambda Ln(REE) Function <pandaslambdas.html>`__
