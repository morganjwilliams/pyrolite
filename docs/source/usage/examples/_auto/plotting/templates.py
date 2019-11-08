"""
Plot Templates
============================

:mod:`pyrolite` includes some ready-made templates for well-known plots. These can
be used to create new plots, or add a template to an existing
:class:`matplotlib.axes.Axes`.
"""
import matplotlib.pyplot as plt
from pyrolite.util.plot import share_axes

# sphinx_gallery_thumbnail_number = 2

########################################################################################
# First let's build a simple TAS diagram:
#
from pyrolite.plot.templates import TAS

ax = TAS()
########################################################################################
# The other templates currently included in pyrolite are the Pearce Th-Nb-Yb and
# Ti-Nb-Yb diagrams. We can create some axes and add these templates to them:
#
from pyrolite.plot.templates import pearceThNbYb, pearceTiNbYb

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
share_axes(ax, which="x")  # these diagrams have the same x axis

pearceThNbYb(ax=ax[0])
pearceTiNbYb(ax=ax[1])

plt.tight_layout()  # nicer spacing for axis labels
