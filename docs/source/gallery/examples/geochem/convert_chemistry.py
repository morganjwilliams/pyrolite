"""
Element-Oxide Transformation
============================

One of pyrolite's strengths is converting mixed elemental and oxide data to a new
form. The simplest way to perform this is by using the
:func:`~pyrolite.geochem.transform.convert_chemistry` function. Note that by default
pyrolite assumes that data are in the same units.
"""
import pyrolite.geochem
import pandas as pd

pd.set_option("precision", 3)  # smaller outputs
########################################################################################
# Here we create some synthetic data to work with, which has some variables in Wt% and
# some in ppm. Notably some elements are present in more than one column (Ca, Na):
#
from pyrolite.util.synthetic import normal_frame

df = normal_frame(columns=["MgO", "SiO2", "FeO", "CaO", "Na2O", "Te", "K", "Na"]) * 100
df.pyrochem.elements *= 100  # elements in ppm
########################################################################################
df.head(2)
########################################################################################
# As the units are heterogeneous, we'll need to convert the data frame to a single set of
# units (here we use Wt%):
#
df.pyrochem.elements = df.pyrochem.elements.pyrochem.scale("ppm", "wt%")  # ppm to wt%
########################################################################################
# We can transform this chemical data to a new set of compositional variables.
# Here we i) convert CaO to Ca, ii) aggregate Na2O and Na to Na and iii) calculate
# mass ratios for Na/Te and MgO/SiO2.
# Note that you can also use this function to calculate mass ratios:
#
df.pyrochem.convert_chemistry(
    to=["MgO", "SiO2", "FeO", "Ca", "Te", "Na", "Na/Te", "MgO/SiO2"]
).head(2)
########################################################################################
# You can also specify molar ratios for iron redox, which will result in multiple iron
# species within the single dataframe:
#
df.pyrochem.convert_chemistry(to=[{"FeO": 0.9, "Fe2O3": 0.1}]).head(2)
