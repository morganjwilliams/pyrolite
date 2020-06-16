"""
Plot Templates
============================

:mod:`pyrolite` includes some ready-made templates for well-known plots. These can
be used to create new plots, or add a template to an existing
:class:`matplotlib.axes.Axes`.
"""
import matplotlib.pyplot as plt
from pyrolite.util.plot.axes import share_axes

# sphinx_gallery_thumbnail_number = 2

########################################################################################
# First let's build a simple total-alkali vs silica (
# :func:`~pyrolite.plot.templates.TAS`) diagram:
#
from pyrolite.plot.templates import TAS

ax = TAS(linewidth=0.5, labels='ID')
plt.show()
########################################################################################
# The other templates currently included in :mod:`pyrolite` are the
# :func:`~pyrolite.plot.templates.pearceThNbYb` and
# :func:`~pyrolite.plot.templates.pearceTiNbYb` diagrams.
# We can create some axes and add these templates to them:
#
from pyrolite.plot.templates import pearceThNbYb, pearceTiNbYb

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
share_axes(ax, which="x")  # these diagrams have the same x axis

pearceThNbYb(ax=ax[0])
pearceTiNbYb(ax=ax[1])

plt.tight_layout()  # nicer spacing for axis labels

########################################################################################
# References and other notes for diagram templates can be found within the docstrings
# and within the pyrolite documentation:
#
help(TAS)
