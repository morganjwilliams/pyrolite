"""
Plot Templates
============================

:mod:`pyrolite` includes some ready-made templates for well-known plots. These can
be used to create new plots, or add a template to an existing
:class:`matplotlib.axes.Axes`.
"""

import matplotlib.pyplot as plt

########################################################################################
# Bivariate Templates
# ~~~~~~~~~~~~~~~~~~~~
# First let's build a simple total-alkali vs silica (
# :func:`~pyrolite.plot.templates.TAS`) diagram:
#
from pyrolite.plot.templates import TAS, SpinelFeBivariate
from pyrolite.util.plot.axes import share_axes

# sphinx_gallery_thumbnail_number = 4


ax = TAS(linewidth=0.5, add_labels=True)
plt.show()
########################################################################################
# A few different variants are now available, with slightly different positioning of
# field boundaries, and with some fields combined:
#
fig, ax = plt.subplots(1, 3, figsize=(12, 3))

TAS(ax=ax[0], linewidth=0.5, add_labels=True, which_model=None)  # Middlemost's TAS
TAS(ax=ax[1], linewidth=0.5, add_labels=True, which_model="LeMaitre")  # LeMaitre's TAS
TAS(ax=ax[2], linewidth=0.5, add_labels=True, which_model="LeMaitreCombined")

for a in ax[1:]:
    a.set(yticks=[], ylabel=None)
########################################################################################
# For distinguishing Fe-rich variants of spinel phases, the bivariate spinel
# diagram can be useful:
#
ax = SpinelFeBivariate(linewidth=0.5, add_labels=True)
plt.show()
########################################################################################
# pyrolite contains templates for the Pearce diagrams, used to discriminate mafic rocks
# (and particularly basalts) based on their whole-rock geochemistry. Two templates are
# included: :func:`~pyrolite.plot.templates.pearceThNbYb` and
# :func:`~pyrolite.plot.templates.pearceTiNbYb`.
# We can create some axes and add these templates to them:
#
from pyrolite.plot.templates import pearceThNbYb, pearceTiNbYb

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
share_axes(ax, which="x")  # these diagrams have the same x axis

pearceThNbYb(ax=ax[0])
pearceTiNbYb(ax=ax[1])

plt.tight_layout()  # nicer spacing for axis labels
########################################################################################
# pyrolite also now includes some diagram templates for discrimination of sandstones
# based on their whole-rock geochemistry (:func:`~pyrolite.plot.templates.Pettijohn`,
# :func:`~pyrolite.plot.templates.Herron`):
#
from pyrolite.plot.templates import Herron, Pettijohn

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
share_axes(ax, which="x")  # these diagrams have the same x axis

Pettijohn(ax=ax[0], add_labels=True)
Herron(ax=ax[1], add_labels=True)

plt.tight_layout()
########################################################################################
# Ternary Templates
# ~~~~~~~~~~~~~~~~~~
# pyrolite now also includes ternary classification diagrams inlcuding
# the :func:`~pyrolite.plot.templates.QAP` and
# :func:`~pyrolite.plot.templates.USDASoilTexture` diagrams:
#
from pyrolite.plot.templates import (
    QAP,
    FeldsparTernary,
    JensenPlot,
    SpinelTrivalentTernary,
    USDASoilTexture,
)

ax = QAP(linewidth=0.4)
plt.show()
########################################################################################
ax = USDASoilTexture(linewidth=0.4)
plt.show()
########################################################################################
# For the feldspar ternary diagram, which is complicated by a miscibility gap, there are
# two modes: `'default'` and `'miscibility-gap'`. The second of these provides a
# simplified approximation of the miscibility gap between k-feldspar and plagioclase,
# wheras 'default' ignores this aspect (which itself is complicated by temperature):
#
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
FeldsparTernary(ax=ax[0], linewidth=0.4, add_labels=True, mode="default")
FeldsparTernary(ax=ax[1], linewidth=0.4, add_labels=True, mode="miscibility-gap")
plt.tight_layout()
plt.show()
########################################################################################
# For general spinel phase discrimination, a ternary classification diagram can
# be used to give labels based on trivalent cationic content (:math:`\mathrm{Cr^{3+}}`,
# :math:`\mathrm{Al^{3+}}`, :math:`\mathrm{Fe^{3+}}`):
#
SpinelTrivalentTernary(linewidth=0.4, add_labels=True, figsize=(6, 6))
plt.show()
########################################################################################
# The Jensen plot is another cationic ternary discrimination diagram (Jensen, 1976),
# for subalkaline volcanic rocks:
#
JensenPlot(linewidth=0.4, add_labels=True, figsize=(7, 7))
plt.show()
########################################################################################
# References and other notes for diagram templates can be found within the docstrings
# and within the pyrolite documentation:
#
help(TAS)
########################################################################################
# .. seealso::
#
#   Examples:
#     `TAS Classifier <../util/TAS.html>`__,
#     `Ternary Colour Mapping <ternary_color.html>`__
#
#   Modules:
#     :mod:`pyrolite.util.classification`
#
#   Classes:
#     :class:`~pyrolite.util.classification.TAS`,
#     :class:`~pyrolite.util.classification.QAP`,
#     :class:`~pyrolite.util.classification.USDASoilTexture`
#
