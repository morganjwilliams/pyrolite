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
# First we'll generate some example synthetic data based around Depleted MORB Mantle:
#
from pyrolite.util.synthetic import example_spider_data

df = example_spider_data(
    noise_level=0.05,
    size=100,
    start="DMM_WH2005",
    norm_to="Chondrite_PON",
    offsets={"Eu": 0.2},
).pyrochem.denormalize_from("Chondrite_PON")
########################################################################################
# Let's have a quick look at what this REE data looks like normalized to Primitive
# Mantle:
#
df.pyrochem.normalize_to("PM_PON").pyroplot.REE(alpha=0.05, c="k", unity_line=True)
plt.show()
########################################################################################
# From this REE data we can fit a series of orthogonal polynomials, and subsequently used
# the regression coefficients ('lambdas') as a parameterisation of the REE
# pattern/profile:
#
ls = df.pyrochem.lambda_lnREE(degree=4)
ls.head(2)
########################################################################################
# So what's actually happening here? To get some idea of what these λ coefficients
# correspond to, we can pull this process apart and visualise our REE profiles as
# the sum of the series of orthogonal polynomial components of increasing order.
# As lambdas represent the coefficients for the regression of log-transformed normalised
# data, to compare the polynomial components and our REE profile we'll first need to
# normalize it to the appropriate composition (here :code:`"ChondriteREE_ON"`) before
# taking the logarithm.
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
from pyrolite.util.lambdas.plot import plot_lambdas_components

ax = (
    df.pyrochem.normalize_to("ChondriteREE_ON")
    .iloc[-1, :]
    .apply(np.log)
    .pyroplot.REE(color="k", label="Data", logy=False)
)

plot_lambdas_components(ls.iloc[-1, :], ax=ax)

ax.legend()
plt.show()
########################################################################################
# Now that we've gone through a brief introduction to how the lambdas are generated,
# let's quickly check what the coefficient values themselves look like:
#

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for ix in range(ls.columns.size - 1):
    ls[ls.columns[ix : ix + 2]].pyroplot.scatter(ax=ax[ix], alpha=0.1, c="k")

plt.tight_layout()
########################################################################################
# But what do these parameters correspond to? From the deconstructed orthogonal
# polynomial above, we can see that :math:`\lambda_0` parameterises relative enrichment
# (this is the mean value of the logarithm of Chondrite-normalised REE abundances),
# :math:`\lambda_1` parameterises a linear slope (here, LREE enrichment), and higher
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
# distributed, so the values themeselves here are not particularly revealing,
# but they do illustrate the expected mangitudes of values for each of the parameters.
#

########################################################################################
# Dealing With Anomalies
# ~~~~~~~~~~~~~~~~~~~~~~~
# Note that we've not used Eu in this regression - Eu anomalies are a deviation from
# the 'smooth profile' we need to use this method. Consider this if your data might also
# exhibit significant Ce anomalies, you might need to exclude this data. For convenience
# there is also functionality to calculate anomalies derived from the orthogonal
# polynomial fit itself (rather than linear interpolation methods). Below we use the
# :code:`anomalies` keyword argument to also calculate the :math:`\frac{Ce}{Ce*}`
# and :math:`\frac{Eu}{Eu*}` anomalies (note that these are excluded from the fit):
#
ls_anomalies = df.pyrochem.lambda_lnREE(anomalies=["Ce", "Eu"])
ax = ls_anomalies.iloc[:, -2:].pyroplot.scatter()
plt.show()
########################################################################################
# Fitting Tetrads
# ~~~~~~~~~~~~~~~~
#
# In addition to fitting orothogonal polynomial functions, the ability to fit tetrad
# functions has also recently been added. This supplements the :math:`\lambda`
# coefficients with :math:`\tau` coefficients which describe subtle electronic
# configuration effects affecting sequential subsets of the REE. Below we plot four
# profiles - each describing only a single tetrad - to illustrate the shape of
# these function components. Note that these are functions of :math:`z`, and are here
# transformed to plot against radii.
#
from pyrolite.util.lambdas.plot import plot_profiles

# let's first create some synthetic pattern parameters
# we want lambdas to be zero, and each of the tetrads to be shown in only one pattern
lambdas = np.zeros((4, 5))
tetrads = np.eye(4)
# putting it together to generate four sets of combined parameters
fit_parameters = np.hstack([lambdas, tetrads])

ax = plot_profiles(
    fit_parameters,
    tetrads=True,
    color=np.arange(4),
)
plt.show()
########################################################################################
# In order to also fit these function components, you can pass the keyword argument
# :code:`fit_tetrads=True` to :func:`~pyrolite.geochem.pyrochem.lambda_lnREE` and
# related functions:
#
lts = df.pyrochem.lambda_lnREE(degree=4, fit_tetrads=True)
########################################################################################
# We can see that the four extra :math:`\tau` Parameters have been appended to the
# right of the lambdas within the output:
#
lts.head(2)
########################################################################################
# More Advanced Customisation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Above we've used default parameterisations for calculating `lambdas`, but
# :mod:`pyrolite` allows you to customise the parameterisation of both the orthogonal
# polynomial components used in the fitting process as well as what data and algorithm
# is used in the fit itself.
#
# To exclude some elements from the *fit* (e.g. Eu which is excluded by default, and
# and potentially Ce), you can either i) filter the dataframe such that the columns
# aren't passed, or ii) explicitly exclude them:
#

# filtering the dataframe first
target_columns = [i for i in df.columns if i not in ["Eu", "Ce"]]
ls_noEuCe_filtered = df[target_columns].pyrochem.lambda_lnREE()
# excluding some elements
ls_noEuCe_excl = df.pyrochem.lambda_lnREE(exclude=["Eu", "Ce"])

# quickly checking equivalence
np.allclose(ls_noEuCe_excl, ls_noEuCe_filtered)
########################################################################################
# While the results should be numerically equivalent, :mod:`pyrolite` does provide
# two algorithms for fitting lambdas. The first follows almost exactly the original
# formulation (:code:`algorithm="ONeill"`; this was translated from VBA), while the
# second simply uses a numerical optimization routine from :mod:`scipy` to achieve the
# same thing (:code:`algorithm="opt"`; this is a fallback for where singular matricies
# pop up):
#

# use the original version
ls_linear = df.pyrochem.lambda_lnREE(algorithm="ONeill")

# use the optimization algorithm
ls_opt = df.pyrochem.lambda_lnREE(algorithm="opt")
########################################################################################
# To quickly demonstrate the equivalance, we can check numerically (to within
# 0.001%):
#
np.allclose(ls_linear, ls_opt, rtol=10e-5)
########################################################################################
# Or simply plot the results from both:
#
fig, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Comparing $\lambda$ Estimation Algorithms", y=1.1)
ls_linear.iloc[:, 1:3].pyroplot.scatter(
    ax=ax, marker="s", c="k", facecolors="none", s=50, label="Linear Algebra"
)
ls_opt.iloc[:, 1:3].pyroplot.scatter(
    ax=ax, c="purple", marker="x", s=50, label="Optimization"
)
ax.legend()
plt.show()
########################################################################################
# You can also use orthogonal polynomials defined over a different set of REE,
# by specifying the parameters using the keyword argument `params`:
#

# this is the original formulation from the paper, where Eu is excluded
ls_original = df.pyrochem.lambda_lnREE(params="ONeill2016")

# this uses a full set of REE
ls_fullREE_polynomials = df.pyrochem.lambda_lnREE(params="full")
########################################################################################
# Note that as of :mod:`pyrolite` v0.2.8, the oringinal formulation is used by default,
# but this will cease to be the case as of the following version, where the full set of
# REE will instead be used to generate the orthogonal polynomials.
#
# While the results are simlar, there are small differences. They're typically less
# than 1%:
np.abs((ls_original / ls_fullREE_polynomials) - 1).max() * 100
########################################################################################
# This can also be visualised:
#
fig, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Comparing Orthogonal Polynomial Bases", y=1.1)
ls_original.iloc[:, 1:3].pyroplot.scatter(
    ax=ax, marker="s", c="k", facecolors="none", s=50, label="Excluding Eu"
)
ls_fullREE_polynomials.iloc[:, 1:3].pyroplot.scatter(
    ax=ax, c="purple", marker="x", s=50, label="All REE"
)
ax.legend()
plt.show()
########################################################################################
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
#    `REE Radii Plot <../plotting/REE_v_radii.html>`__
#
#   Functions:
#     :func:`~pyrolite.geochem.pyrochem.lambda_lnREE`,
#     :func:`~pyrolite.geochem.ind.get_ionic_radii`,
#     :func:`pyrolite.plot.pyroplot.REE`
#
