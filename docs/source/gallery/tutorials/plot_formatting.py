"""
Formatting and Cleaning Up Plots
==================================

.. note:: This tutorial is a work in progress and will be gradually updated.

In this tutorial we will illustrate some straightfoward formatting for your plots which
will allow for greater customisation as needed. As :mod:`pyrolite` heavily uses
and exposes the API of :mod:`matplotlib` for the visualisation components
(and also :mod:`mpltern` for ternary diagrams), you should also check out their
documentation pages for more in-depth guides, examples and API documentation.
"""
#######################################################################################
# First let's pull in a simple dataset to use throughout these examples:
#
from pyrolite.util.synthetic import normal_frame

# sphinx_gallery_thumbnail_number = 7

df = normal_frame(columns=["SiO2", "CaO", "MgO", "Al2O3", "TiO2", "27Al", "d11B"])
#######################################################################################
# Basic Figure and Axes Settings
# ------------------------------
#
# :mod:`matplotlib` makes it relatively straightfoward to customise most settings for
# your figures and axes. These settings can be defined at creation (e.g. in a call to
# :func:`~matplotlib.pyplot.subplots`), or they can be defined after you've created an
# axis (with the methods :code:`ax.set_<parameter>()`). For example:
#
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)

ax.set_xlabel("My X Axis Label")
ax.set_title("My Axis Title", fontsize=12)
ax.set_yscale("log")
ax.set_xlim((0.5, 10))

fig.suptitle("My Figure Title", fontsize=15)

plt.show()
#######################################################################################
# You can use a single method to set most of these things:
# :func:`~matplotlib.axes.Axes.set`. For example:
#
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
ax.set(yscale="log", xlim=(0, 1), ylabel="YAxis", xlabel="XAxis")
plt.show()
#######################################################################################
# Labels and Text
# ----------------
#
# :mod:`matplotlib` enables you to use :math:`\TeX` within all text elements, including
# labels and annotations. This can be leveraged for more complex formatting,
# incorporating math and symbols into your plots. Check out the mod:`matplotlib`
# `tutorial <https://matplotlib.org/3.2.1/tutorials/text/mathtext.html>`__, and
# for more on working with text generally in :mod:`matplotlib`, check out the
# `relevant tutorials gallery <https://matplotlib.org/3.2.1/tutorials/index.html#text>`__.
#
# The ability to use TeX syntax in :mod:`matplotlib` text objects can also be used
# for typsetting, like for subscripts and superscripts. This is particularly relevant
# for geochemical oxides labels (e.g. Al2O3, which would ideally be rendered as
# :math:`Al_2O_3`) and isotopes (e.g. d11B, which should be :math:`\delta^{11}B`).
# At the moment, pyrolite won't do this for you, so you may want to adjust the labelling
# after you've made them. For example:
import pyrolite.plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)
df[["Al2O3", "TiO2"]].pyroplot.scatter(ax=ax[0])
ax[0].set_xlabel("Al$_2$O$_3$")
ax[0].set_ylabel("TiO$_2$")

df[["27Al", "d11B"]].pyroplot.scatter(ax=ax[1])
ax[1].set_xlabel("$^{27}$Al")
ax[1].set_ylabel("$\delta^{11}$B")

plt.tight_layout()  # rearrange the plots to fit nicely together
plt.show()
#######################################################################################
# Sharing Axes
# ------------
#
# If you're building figures which have variables which are re-used, you'll typically
# want to 'share' them between your axes. The :mod:`matplotlib.pyplot` API makes
# this easy for when you want to share among *all* the axes as your create them:
#
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
plt.show()
#######################################################################################
# However, if you want to share axes in a way which is less standard, it can be
# difficult to set up using this function. :mod:`pyrolite` has a utility function
# which can be used to share axes after they're created in slightly more arbitrary
# ways. For example, imagine we wanted to share the first and third x-axes, and the
# first three y-axes, you could use:
#
import matplotlib.pyplot as plt
from pyrolite.util.plot.axes import share_axes

fig, ax = plt.subplots(2, 2)
ax = ax.flat  # turn the (2,2) array of axes into one flat axes with shape (4,)
share_axes([ax[0], ax[2]], which="x")  # share x-axes for 0, 2
share_axes(ax[0:3], which="y")  # share y-axes for 0, 1, 2

ax[0].set_xlim((0, 10))
ax[1].set_ylim((-5, 5))
plt.show()
#######################################################################################
# Legends
# -------
#
# While it's simple to set up basic legends in :mod:`maplotlib` (see the docs for
# :func:`matplotlib.axes.Axes.legend`), often you'll want to customise
# your legends to fit nicely within your figures. Here we'll create a few
# synthetic datasets, add them to a figure and create the default legend:
import pyrolite.plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
for i in range(3):
    sample_data = normal_frame(columns=["CaO", "MgO", "FeO"])  # a new random sample
    sample_data[["CaO", "MgO"]].pyroplot.scatter(ax=ax, label="Sample {:d}".format(i))
ax.legend()
plt.show()
#######################################################################################
# On many of the :mod:`pyrolite` examples, you'll find legends formatted along the
# lines of the following to clean them up a little (these are the default styles):
#
ax.legend(
    facecolor=None,  # have a transparent legend background
    frameon=False,  # remove the legend frame
    bbox_to_anchor=(1, 1),  # anchor legend's corner to the axes' top-right
    loc="upper left",  # use the upper left corner for the anchor
)
plt.show()
#######################################################################################
# Check out the :mod:`matplotlib`
# `legend guide <https://matplotlib.org/tutorials/intermediate/legend_guide.html>`__
# for more.
#
# Ternary Plots
# -------------
#
# The ternary plots in :mod:`pyrolite` are generated using :mod:`mpltern`, and while
# the syntax is very similar to the :mod:`matplotlib` API, as we have three axes
# to deal with sometimes things are little different. Here we demonstrate how to
# complete some common tasks, but you should check out the :mod:`mpltern` documentation
# if you want to dig deeper into customising your ternary diagrams (e.g. see the
# `example gallery <https://mpltern.readthedocs.io/en/latest/gallery/index.html>`__),
# which these examples were developed from.
#
# One of the key things to note in :mod:`mpltern` is that you have `top`, `left` and
# `right` axes.
#
# Ternary Plot Axes Labels
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Labelling ternary axes is done similarly to in :mod:`matplotlib`, but using the
# axes prefixes `t`, `l` and `r` for top, left and right axes, respectively:
#
import pyrolite.plot
import matplotlib.pyplot as plt

ax = df[["CaO", "MgO", "Al2O3"]].pyroplot.scatter()
ax.set_tlabel("Top")
ax.set_llabel("Left")
ax.set_rlabel("Right")
plt.show()
########################################################################################
# Ternary Plot Grids
# ~~~~~~~~~~~~~~~~~~
#
# To add a simple grid to your ternary plot, you can use
# :func:`~mpltern.TernaryAxis.grid`:
#
import pyrolite.plot
import matplotlib.pyplot as plt

ax = df[["CaO", "MgO", "Al2O3"]].pyroplot.scatter()
ax.grid()
plt.show()
########################################################################################
# With this method, you can also specify an `axis`, `which` tickmarks you want to use
# for the grid ('major', 'minor' or 'both') and a `linestyle`:
#
import pyrolite.plot
import matplotlib.pyplot as plt

ax = df[["CaO", "MgO", "Al2O3"]].pyroplot.scatter()
ax.grid(axis="r", which="both", linestyle="--")
plt.show()
########################################################################################
# Ternary Plot Limits
# ~~~~~~~~~~~~~~~~~~~
#
# To focus on a specific area, you can reset the limits of your ternary axes with
# :func:`~mpltern.TernaryAxis.set_ternary_lim`.
#
# Also check out the :mod:`mpltern`
# `inset axes example <https://mpltern.readthedocs.io/en/latest/gallery/advanced/05.inset.html>`__
# if you're after ways to focus on specific regions.
#
import pyrolite.plot
import matplotlib.pyplot as plt

ax = df[["CaO", "MgO", "Al2O3"]].pyroplot.scatter()
ax.set_ternary_lim(
    0.1, 0.5, 0.2, 0.6, 0.3, 0.7  # tmin  # tmax  # lmin  # lmax  # rmin  # rmax
)
plt.show()
