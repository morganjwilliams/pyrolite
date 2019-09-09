from pyrolite.geochem.transform import convert_chemistry

# %% Random data
import numpy as np
import pandas as pd
from pyrolite.util.synthetic import test_df

np.random.seed(82)
df = (test_df(cols=["MgO", "SiO2", "FeO", "CaO", "Na2O"]) * 100).join(
    test_df(cols=["Ca", "Te", "K", "Na"]) * 1000
)
# %% Unit Conversion
from pyrolite.geochem.ind import common_elements
from pyrolite.geochem.norm import scale

ppm_cols = [i for i in df.columns if i in common_elements()]  # elemental headers
df.loc[:, ppm_cols] *= scale("ppm", "wt%")
# %% Conversion
new_df = convert_chemistry(
    df, to=["MgO", "SiO2", "FeO", "CaO", "Te", "Na", "Na/Te", "MgO/SiO2"]
)
ppm_cols = [i for i in new_df.columns if i in common_elements()]  # elemental headers
df.loc[:, ppm_cols] *= scale("wt%", "ppm") # convert Te, Na to ppm
# %% Iron Conversion
new_df = convert_chemistry(df, to=["MgO", "SiO2", {"FeO": 0.9, "Fe2O3": 0.1}, "CaO"])
