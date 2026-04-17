"""
Ternary Color Systems
============================

:mod:`pyrolite` includes two methods for coloring data points and polygons in
a ternary system, :func:`~pyrolite.util.plot.style.ternary_color` and
:func:`~pyrolite.util.plot.style.color_ternary_polygons_by_centroid` which work
well with some of the plot templates (:mod:`pyrolite.plot.templates`) and
associated classifiers (:mod:`pyrolite.util.classification`).
"""

import numpy as np
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 5

########################################################################################
# Colors by Ternary Position
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The :func:`~pyrolite.util.plot.style.ternary_color` function serves to generate
# the color which correpsonds the the mixing of three colours in proportion
# to the components in a ternary system. By default these colours are red, green
# and blue (corresponding to the top, left, and right components in a terarny diagram).
# The function returns colours in the form of an RGBA array:
#
from pyrolite.util.plot.style import ternary_color
from pyrolite.util.synthetic import normal_frame

# generate a synthetic dataset we can use for the colouring example
df = normal_frame(
    columns=["CaO", "MgO", "FeO"],
    size=100,
    seed=42,
    cov=np.array([[0.8, 0.3], [0.3, 0.8]]),
)

colors = ternary_color(df)
colors[:3]
########################################################################################
# These can then be readily used in a ternary diagram (or eleswhere):
#
ax = df.pyroplot.scatter(c=colors)
plt.show()
########################################################################################
ax = df[["MgO", "CaO"]].pyroplot.scatter(c=colors)
plt.show()
########################################################################################
# You can use different colors for each of the verticies if you so wish, and
# mix and match named colors with RGB/RGBA represntations (note that the alpha will
# be scaled, if it is passed as a keyword argument to
# :func:`~pyrolite.util.plot.style.ternary_color`):
#
colors = ternary_color(df, alpha=0.9, colors=["green", "orange", [0.9, 0.1, 0.5, 0.9]])
ax = df.pyroplot.scatter(c=colors)
plt.show()
########################################################################################
# Colors by Centroid Position
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can also colour polygons within one of these templates by the ternary
# combination of colours (defaulting to red, green and blue) at the polygon
# centroid:
#
from pyrolite.util.classification import USDASoilTexture
from pyrolite.util.plot.style import color_ternary_polygons_by_centroid

clf = USDASoilTexture()
ax = clf.add_to_axes(ax=None, add_labels=True, figsize=(8, 8))
color_ternary_polygons_by_centroid(ax)
plt.show()
########################################################################################
# There are a range of options you can pass to this function to control the
# ternary colors (as above), change the scaling coefficients for ternary components
# and change the opacity of the colors:
#
color_ternary_polygons_by_centroid(
    ax, colors=("red", "green", "blue"), coefficients=(1, 1, 1), alpha=0.5
)
plt.show()
########################################################################################
# .. seealso::
#
#   Examples:
#     `Ternary Diagrams <ternary.html>`__,
#     `Plot Templates <templates.html>`__
#
