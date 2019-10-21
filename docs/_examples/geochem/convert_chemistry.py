import pyrolite.geochem

# %% Random data
import numpy as np
import pandas as pd
from pyrolite.util.synthetic import test_df

np.random.seed(82)
df = (test_df(cols=["MgO", "SiO2", "FeO", "CaO", "Na2O"]) * 100).join(
    test_df(cols=["Ca", "Te", "K", "Na"]) * 1000
)
# %% Unit Conversion
from pyrolite.geochem.norm import scale

df.loc[:, df.pyrochem.elements] *= scale("ppm", "wt%")
# %% Conversion
new_df = df.pyrochem.convert_chemistry(
    to=["MgO", "SiO2", "FeO", "CaO", "Te", "Na", "Na/Te", "MgO/SiO2"]
)
df.loc[:, new_df.pyrochem.elements] *= scale("wt%", "ppm")  # convert Te, Na to ppm
# %% Iron Conversion
new_df = df.pyrochem.convert_chemistry(
    to=["MgO", "SiO2", {"FeO": 0.9, "Fe2O3": 0.1}, "CaO"]
)
