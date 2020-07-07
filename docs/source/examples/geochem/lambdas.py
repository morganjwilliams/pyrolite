"""
lambdas: Parameterising REE Profiles
=====================================

Orthogonal polynomial decomposition can be used for dimensional reduction of smooth
function over an independent variable, producing an array of independent values
representing the relative weights for each order of component polynomial. This is an
effective method to parameterise and compare the nature of smooth profiles.

In geochemistry, the most applicable use case is for reduction Rare Earth Element (REE)
profiles. The REE are a collection of elements with broadly similar physicochemical
properties (the lanthanides), which vary with ionic radii. Given their similar behaviour
and typically smooth function of normalised abundance vs. ionic radii, the REE profiles
and their shapes can be effectively parameterised and dimensionally reduced (14 elements
summarised by 3-4 shape parameters).

Here we generate some example data, reduce these to lambda values, and visualise the
results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyrolite.plot

# sphinx_gallery_thumbnail_number = 2

np.random.seed(82)
########################################################################################
# First we'll generate some example synthetic data based around Depleted Morb Mantle:
#
from pyrolite.util.synthetic import example_spider_data

df = example_spider_data(
    noise_level=0.05,
    nobs=100,
    start="DMM_WH2005",
    norm_to="Chondrite_PON",
    offsets={"Eu": 0.2},
)
########################################################################################
# Let's have a quick look at what this REE data looks like:
#
df.pyroplot.REE(alpha=0.05, c="k", unity_line=True)
plt.show()
########################################################################################
# From this REE data we can fit a series of orthogonal polynomials, and subsequently used
# the regression coefficients ('lambdas') as a parameterisation of the REE
# pattern/profile. This example data is already normalised to Chondrite, so to avoid
# double-normalising, we pass :code:`norm_to=None`:
#
ls = df.pyrochem.lambda_lnREE(degree=4, norm_to=None)
########################################################################################
# So what's actually happening here? To get some idea of what these λ coefficients
# correspond to, we can pull this process apart and visualse our REE profiles as
# the sum of the series of orthogonal polynomial components of increasing order.
# As lambdas represent the coefficients for the regression of log-transformed normalised
# data, we'll first need to take the logarithm.
#
# With our data, we've then fit a function of ionic radius with the form
# :math:`f(r) = \lambda_0 + \lambda_1 f_1 + \lambda_2 f_2 + \lambda_3 f_3...`
# where the polynomial components of increasing order are :math:`f_1 = (r - \beta_0)`,
# :math:`f_2 = (r - \gamma_0)(r - \gamma_1)`,
# :math:`f_3 = (r - \delta_0)(r - \delta_1)(r - \delta_2)` and so on. The parameters
# :math:`\beta`, :math:`\gamma`, :math:`\delta` are pre-computed such that the
# polynomial components are indeed independent. Here we can visualise how these
# polynomial components are summed to produce the regressed profile, using the last REE
# profile we generated above as an example:
#
from pyrolite.util.lambdas import plot_lambdas_components

ax = df.iloc[-1, :].apply(np.log).pyroplot.REE(color="k", label="Data", logy=False)

plot_lambdas_components(ls.iloc[-1, :], ax=ax)

ax.legend(frameon=False, facecolor=None, bbox_to_anchor=(1, 1))
plt.show()
########################################################################################
# Note that we've not used Eu in this regression - Eu anomalies are a deviation from
# the 'smooth profile' we need to use this method. Consider this if your data might also
# exhibit significant Ce anomalies, you might need to exclude this data.
#
# Now that we've gone through a brief introduction to how the lambdas are generated,
# let's quickly check what the coefficient values themselves look like:
#

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for ix in range(ls.columns.size - 1):
    ls[ls.columns[ix : ix + 2]].pyroplot.scatter(ax=ax[ix], alpha=0.1, c="k")

plt.tight_layout()
########################################################################################
# But what do these parameters correspond to? From the deconstructed orthogonal
# polynomial above, we can see that :math:`\lambda_0` parameterises relative enrichement
# (this is the mean value of the logarithm of Chondrite-normalised REE abundances),
# :math:`\lambda_1` parameterises a linear slope (here, LREE enrichemnt), and higher
# order terms describe curvature of the REE pattern. Through this parameterisation,
# the REE profile can be effectively described and directly linked to geochemical
# processes. While the amount of data we need to describe the patterns is lessened,
# the values themselves are more meaningful and readily used to describe the profiles
# and their physical significance.
#
# The visualisation of :math:`\lambda_1`-:math:`\lambda_2` can be particularly useful
# where you're trying to compare REE profiles.
#
# We've used a synthetic dataset here which is by design approximately normally
# distrtibuted, so the values themeselves here are not particularly revealing,
# but they do illustrate the expected mangitudes of values for each of the parameters.
#
# For more on using orthogonal polynomials to describe geochemical pattern data, dig
# into the paper which introduced the method to geochemists:
# O’Neill, H.S.C., 2016. The Smoothness and Shapes of Chondrite-normalized Rare Earth
# Element Patterns in Basalts. J Petrology 57, 1463–1508.
# `doi: 10.1093/petrology/egw047 <https://doi.org/10.1093/petrology/egw047>`__.
#
# .. seealso::
#
#   Examples:
#    `Ionic Radii <ionic_radii.html>`__,
#    `REE Radii Plot <../plotting/REE_radii_plot.html>`__
#
#   Functions:
#     :func:`~pyrolite.geochem.pyrochem.lambda_lnREE`,
#     :func:`~pyrolite.geochem.ind.get_ionic_radii`,
#     :func:`pyrolite.plot.pyroplot.REE`
#
