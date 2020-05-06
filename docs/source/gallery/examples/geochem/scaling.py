"""
Unit Scaling
=============
"""
import pyrolite.geochem
import pandas as pd

pd.set_option("precision", 3)  # smaller outputs
########################################################################################
from pyrolite.util.synthetic import test_df

df = test_df(cols=['CaO', 'MgO', 'SiO2', 'FeO', 'Ni', 'Ti', 'La', 'Lu', 'Mg/Fe'])
########################################################################################
cols = ["Ni", "NiO", "La", "La2O3"]
df.head(2).pyrochem.convert_chemistry(to=cols)[cols]  # these are in ppm!
