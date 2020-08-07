"""
Reference Compositions
=======================

This page presents the range of compositions within the reference compositon database
accessible within :mod:`pyrolite`. It's currently a work in progress, but will soon
contain extended descriptions and notes for some of the compositions and associated
references.
"""
import matplotlib.pyplot as plt
from pyrolite.geochem.norm import all_reference_compositions, get_reference_composition
# sphinx_gallery_thumbnail_number = 11

refcomps = all_reference_compositions()
norm = "Chondrite_PON"  # a constant composition to normalise to
########################################################################################
# Chondrites
# -----------
#
fltr = lambda c: c.reservoir == "Chondrite"
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Mantle
# -------
#
# Primitive Mantle & Pyrolite
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
fltr = lambda c: c.reservoir in ["PrimitiveMantle", "BSE"]
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Depleted Mantle
# ~~~~~~~~~~~~~~~~
#
fltr = lambda c: ("Depleted" in c.reservoir) & ("Mantle" in c.reservoir)
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Mid-Ocean Ridge Basalts (MORB)
# -------------------------------
#
#
# Average MORB, NMORB
# ~~~~~~~~~~~~~~~~~~~~~
#
fltr = lambda c: c.reservoir in ["MORB", "NMORB"]
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
#
# Enriched MORB
# ~~~~~~~~~~~~~
#
fltr = lambda c: "EMORB" in c.reservoir
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()

########################################################################################
# Ocean Island Basalts
# --------------------
#
fltr = lambda c: "OIB" in c.reservoir
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Continental Crust
# -----------------
#
# Bulk Continental Crust
# ~~~~~~~~~~~~~~~~~~~~~~~
#
fltr = lambda c: c.reservoir == "BulkContinentalCrust"
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Upper Continental Crust
# ~~~~~~~~~~~~~~~~~~~~~~~
#
fltr = lambda c: c.reservoir == "UpperContinentalCrust"
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Mid-Continental Crust
# ~~~~~~~~~~~~~~~~~~~~~
#
fltr = lambda c: c.reservoir == "MidContinentalCrust"
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Lower Continental Crust
# ~~~~~~~~~~~~~~~~~~~~~~~
#
fltr = lambda c: c.reservoir == "LowerContinentalCrust"
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Shales
# ------
#
fltr = lambda c: "Shale" in c.reservoir
compositions = [x for (name, x) in refcomps.items() if fltr(x)]

fig, ax = plt.subplots(1)
for composition in compositions:
    composition.set_units("ppm")
    df = composition.comp.pyrochem.normalize_to(norm, units="ppm")
    df.pyroplot.REE(unity_line=True, ax=ax, label=composition.name)
ax.legend()
plt.show()
########################################################################################
# Composition List
# -----------------
#
# |refcomps|
#
# .. seealso::
#
#    Examples:
#     `Normalisation <../examples/geochem/normalization.html>`__
#
