"""
lambdas: Visualising Orthogonal Polynomials
============================================
"""
import numpy as np
from pyrolite.plot.spider import REE_v_radii
from pyrolite.geochem.ind import REE, get_ionic_radii
from pyrolite.util.math import lambdas, lambda_poly_func, OP_constants

np.random.seed(82)


def plot_orthagonal_polynomial_components(ax, xs, lambdas, params, log=False, **kwargs):
    """Plot polynomials on an axis over x values."""
    for w, p in zip(lambdas, params):  # plot the polynomials
        f = np.ones_like(xs) * w
        for c in p:
            f *= xs - np.float(c)
        if log:
            f = np.exp(f)

        label = (
            "$r^{}: \lambda_{}".format(len(p), len(p))
            + ["\cdot f_{}".format(len(p)), ""][int(len(p) == 0)]
            + "$"
        )
        ax.plot(xs, f, label=label, **kwargs)

########################################################################################
# First we generate some example data:
#
data_ree = REE(dropPm=True)
data_radii = np.array(get_ionic_radii(data_ree, charge=3, coordination=8))
lnY = (
    np.random.randn(*data_radii.shape) * 0.1
    + np.linspace(3.0, 0.0, data_radii.size)
    + (data_radii - 1.11) ** 2.0
    - 0.1
)

for ix, el in enumerate(data_ree):
    if el in ["Ce", "Eu"]:
        lnY[ix] += np.random.randn(1) * 0.6

Y = np.exp(lnY)
########################################################################################
# Now we can calculate the lambdas:
#
exclude = ["Ce", "Eu"]
if exclude:
    subset_ree = [i for i in data_ree if not i in exclude]
    subset_Y = Y[[i in subset_ree for i in data_ree]]
    subset_radii = np.array(get_ionic_radii(subset_ree, charge=3, coordination=8))
else:
    subset_Y, subset_ree, subset_radii = Y, data_ree, data_radii

params = OP_constants(subset_radii, degree=4)
ls = lambdas(np.log(subset_Y), subset_radii, params=params, degree=4)
continuous_radii = np.linspace(subset_radii[0], subset_radii[-1], 20)
l_func = lambda_poly_func(ls, pxs=subset_radii, params=params)
smooth_profile = np.exp(l_func(continuous_radii))
########################################################################################
ax = REE_v_radii(Y, ree=data_ree, index="radii", color="0.8", label="Data")
REE_v_radii(
    subset_Y,
    ree=subset_ree,
    ax=ax,
    index="radii",
    color="k",
    linewidth=0,
    label="Subset",
)
plot_orthagonal_polynomial_components(ax, continuous_radii, ls, params, log=True)
ax.plot(continuous_radii, smooth_profile, label="Reconstructed\nProfile", c="k", lw=2)
ax.legend(frameon=False, facecolor=None, bbox_to_anchor=(1, 1))
########################################################################################
# For more on using orthogonal polynomials to describe geochemical pattern data, see:
# O’Neill, H.S.C., 2016. The Smoothness and Shapes of Chondrite-normalized Rare Earth
# Element Patterns in Basalts. J Petrology 57, 1463–1508.
# `doi: 10.1093/petrology/egw047 <https://doi.org/10.1093/petrology/egw047>`__.
#
# .. seealso::
#
#   Examples:
#     `Pandas Lambda Ln(REE) Function <pandaslambdas.html>`__,
#     `Dimensional Reduction <lambdadimreduction.html>`__,
#     `REE Radii Plot <../plotting/REE_v_radii.html>`__
