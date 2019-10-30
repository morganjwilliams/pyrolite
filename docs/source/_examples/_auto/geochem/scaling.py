"""
Unit Scaling
=============
"""
import pyrolite.geochem
########################################################################################
from pyrolite.util.synthetic import test_df

df = test_df(cols=['CaO', 'MgO', 'SiO2', 'FeO', 'Ni', 'Ti', 'La', 'Lu', 'Mg/Fe'])
########################################################################################
cols = ["Ni", "NiO", "La", "La2O3"]
df.head(2).pyrochem.convert_chemistry(to=cols)[cols]  # these are in ppm!
