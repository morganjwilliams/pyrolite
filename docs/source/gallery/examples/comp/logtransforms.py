"""
Log-transforms
----------------

pyrolite includes a few functions for dealing with compositional data, at the heart of
which are i) closure (i.e. everything sums to 100%) and ii) log-transforms to deal with
the compositional space. The commonly used log-transformations include the
Additive Log-Ratio (:func:`~pyrolite.comp.pyrocomp.ALR`), Centred Log-Ratio
(:func:`~pyrolite.comp.pyrocomp.CLR`), and Isometric Log-Ratio
(:func:`~pyrolite.comp.pyrocomp.ILR`) [#ref_1]_ [#ref_2]_.

This example will show you how to access and use some of these functions in pyrolite.
"""

########################################################################################
# First let's create some example data:
#
from pyrolite.util.synthetic import normal_frame, random_cov_matrix

df = normal_frame(
    size=100,
    cov=random_cov_matrix(sigmas=[0.1, 0.05, 0.3, 0.6], dim=4, seed=32),
    seed=32,
)
df.describe()
########################################################################################
# Let's have a look at some of the log-transforms, which can be accessed directly from
# your dataframes (via :class:`pyrolite.comp.pyrocomp`), after you've imported
# :mod:`pyrolite.comp`. Note that the transformations will return *new* dataframes,
# rather than modify their inputs. For example:
#
import pyrolite.comp

lr_df = df.pyrocomp.CLR()  # using a centred log-ratio transformation
########################################################################################
# The transformations are implemented such that the column names generally make it
# evident which transformations have been applied (here using default simple labelling;
# see below for other examples):
#
lr_df.columns
########################################################################################
# To invert these transformations, you can call the respective inverse transform:
#
back_transformed = lr_df.pyrocomp.inverse_CLR()
########################################################################################
# Given we haven't done anything to our dataframe in the meantime, we should be back
# where we started, and our values should all be equal within numerical precision.
# To verify this, we can use :func:`numpy.allclose`:
#
import numpy as np

np.allclose(back_transformed, df)
########################################################################################
# In addition to easy access to the transforms, there's also a convenience function
# for taking a log-transformed mean (log-transforming, taking a mean, and inverse log
# transforming; :func:`~pyrolite.comp.codata.pyrocomp.logratiomean`):
#

df.pyrocomp.logratiomean()
########################################################################################
# While this function defaults to using :func:`~pyrolite.comp.codata.clr`,
# you can specify other log-transforms to use:
#
df.pyrocomp.logratiomean(transform="CLR")
########################################################################################
# Notably, however, the logratio means should all give you the same result:
#
np.allclose(
    df.pyrocomp.logratiomean(transform="CLR"),
    df.pyrocomp.logratiomean(transform="ALR"),
) & np.allclose(
    df.pyrocomp.logratiomean(transform="CLR"),
    df.pyrocomp.logratiomean(transform="ILR"),
)
########################################################################################
# To change the default labelling outputs for column names, you can use the
# `label_mode` parameter, for example to get nice labels for plotting:
#
import matplotlib.pyplot as plt
df.pyrocomp.ILR(label_mode="latex").iloc[:, 0:2].pyroplot.scatter()
plt.show()
########################################################################################
# Alternatively if you simply want numeric indexes which you can use in e.g. a ML
# pipeline, you can use :code:`label_mode="numeric"`:
df.pyrocomp.ILR(label_mode="numeric").columns
########################################################################################
# .. [#ref_1] Aitchison, J., 1984. The statistical analysis of geochemical compositions.
#       Journal of the International Association for Mathematical Geology 16, 531–564.
#       doi: `10.1007/BF01029316 <https://doi.org/10.1007/BF01029316>`__
#
# .. [#ref_2]  Egozcue, J.J., Pawlowsky-Glahn, V., Mateu-Figueras, G.,
#       Barceló-Vidal, C., 2003.
#       Isometric Logratio Transformations for Compositional Data Analysis.
#       Mathematical Geology 35, 279–300.
#       doi: `10.1023/A:1023818214614 <https://doi.org/10.1023/A:1023818214614>`__
#
# .. seealso::
#
#   Examples:
#     `Log Ratio Means <logratiomeans.html>`__,
#     `Compositional Data <compositional_data.html>`__,
#     `Ternary Plots <../plotting/ternary.html>`__
#
#   Tutorials:
#     `Ternary Density Plots <../../tutorials/ternary_density.html>`__,
#     `Making the Logo <../../tutorials/logo.html>`__
#
#   Modules and Functions:
#     :mod:`pyrolite.comp.codata`,
#     :func:`~pyrolite.comp.codata.boxcox`,
#     :func:`~pyrolite.comp.pyrocomp.renormalise`
