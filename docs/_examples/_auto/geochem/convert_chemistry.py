"""
Convert Chemistry
==================
"""
import pyrolite.geochem
########################################################################################
from pyrolite.util.synthetic import test_df

df = test_df(cols=['CaO', 'MgO', 'SiO2', 'FeO', 'Mn', 'Ti', 'La', 'Lu', 'Mg/Fe'])
########################################################################################
lets_get = df.pyrochem.list_oxides + df.pyrochem.list_REE + [{"FeO": 0.9, "Fe2O3": 0.1}]
df.head(2).pyrochem.convert_chemistry(to=lets_get)
