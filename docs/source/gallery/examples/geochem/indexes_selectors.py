"""
Geochemical Indexes and Selectors
==================================

"""
import pyrolite.geochem
import pandas as pd

pd.set_option("precision", 3)  # smaller outputs
########################################################################################
from pyrolite.util.synthetic import test_df

df = test_df(cols=["CaO", "MgO", "SiO2", "FeO", "Mn", "Ti", "La", "Lu", "Mg/Fe"])
########################################################################################

df.head(2).pyrochem.oxides

########################################################################################

df.head(2).pyrochem.elements

########################################################################################

df.head(2).pyrochem.REE

########################################################################################

df.head(2).pyrochem.compositional

########################################################################################

df.pyrochem.list_oxides

########################################################################################

df.pyrochem.list_elements

########################################################################################

df.pyrochem.list_REE

########################################################################################

df.pyrochem.list_compositional

########################################################################################
# All elements (up to U):
#
from pyrolite.geochem.ind import common_elements, common_oxides, REE

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
