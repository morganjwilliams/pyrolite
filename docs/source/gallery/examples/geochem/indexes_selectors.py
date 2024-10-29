"""
Geochemical Indexes and Selectors
==================================

"""

import pandas as pd

import pyrolite.geochem

pd.set_option("display.precision", 3)  # smaller outputs
########################################################################################
from pyrolite.util.synthetic import normal_frame

df = normal_frame(
    columns=[
        "CaO",
        "MgO",
        "SiO2",
        "FeO",
        "Mn",
        "Ti",
        "La",
        "Lu",
        "Y",
        "Mg/Fe",
        "87Sr/86Sr",
        "Ar40/Ar36",
    ]
)
########################################################################################

df.head(2).pyrochem.oxides

########################################################################################

df.head(2).pyrochem.elements

########################################################################################

df.head(2).pyrochem.REE

########################################################################################

df.head(2).pyrochem.REY

########################################################################################

df.head(2).pyrochem.compositional

########################################################################################

df.head(2).pyrochem.isotope_ratios

########################################################################################

df.pyrochem.list_oxides

########################################################################################

df.pyrochem.list_elements

########################################################################################

df.pyrochem.list_REE

########################################################################################

df.pyrochem.list_compositional

########################################################################################

df.pyrochem.list_isotope_ratios

########################################################################################
# All elements (up to U):
#
from pyrolite.geochem.ind import REE, REY, common_elements, common_oxides

common_elements()  # string return

########################################################################################
# All elements, returned as a list of `~periodictable.core.Formula`:
#
common_elements(output="formula")  # periodictable.core.Formula return

########################################################################################
# Oxides for elements with positive charges (up to U):
#
common_oxides()

########################################################################################

REE()

########################################################################################
REY()
