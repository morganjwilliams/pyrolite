"""
Unit Scaling
=============
"""

import numpy as np
import pandas as pd

import pyrolite.geochem

pd.set_option("display.precision", 3)  # smaller outputs
########################################################################################
# Here we create an example dataframe to work from, containing some synthetic data
# in the form of oxides and elements, each with different units (wt% and ppm, respectively).
#
from pyrolite.util.synthetic import normal_frame

df = normal_frame(
    columns=["CaO", "MgO", "SiO2", "FeO", "Ni", "Ti", "La", "Lu"], seed=22
)
df.pyrochem.oxides *= 100  # oxides in wt%
df.pyrochem.elements *= 10000  # elements in ppm
########################################################################################
# In this case, we might want to transform the Ni and Ti into their standard oxide
# equivalents NiO and TiO2:
#
df.pyrochem.convert_chemistry(to=["NiO", "TiO2"]).head(2)
########################################################################################
# But here because Ni and Ti have units of ppm, the results are a little non-sensical,
# especially when it's combined with the other oxides:
#
df.pyrochem.convert_chemistry(to=df.pyrochem.list_oxides + ["NiO", "TiO2"]).head(2)
########################################################################################
# There are multiple ways we could convert the units, but here we're going to first
# convert the elemental ppm data to wt%, then perform our oxide-element conversion.
# To do this, we'll use the built-in function :func:`~pyrolite.util.units.scale`:
#
from pyrolite.util.units import scale

df.pyrochem.elements *= scale("ppm", "wt%")
########################################################################################
# We can see that this then gives us numbers which are a bit more sensible:
#
df.pyrochem.convert_chemistry(to=df.pyrochem.list_oxides + ["NiO", "TiO2"]).head(2)
########################################################################################
# Dealing with Units in Column Names
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Often our dataframes will start containing column names which pyrolite doesn't recognize
# natively by default (work in progress, this is an item on the roadmap). Here we can
# create an example of that, and go through some key steps for using this data in
# :mod:`pyrolite`:
#
df = normal_frame(
    columns=["CaO", "MgO", "SiO2", "FeO", "Ni", "Ti", "La", "Lu"], seed=22
)
df.pyrochem.oxides *= 100  # oxides in wt%
df.pyrochem.elements *= 10000  # elements in ppm
df = df.rename(
    columns={
        **{c: c + "_wt%" for c in df.pyrochem.oxides},
        **{c: c + "_ppm" for c in df.pyrochem.elements},
    }
)
df.head(2)
########################################################################################
# If you just wanted to rescale some columns, you can get away without renaming your
# columns,  e.g. converting all of the ppm columns to wt%:
#
df.loc[:, [c for c in df.columns if "_ppm" in c]] *= scale("ppm", "wt%")
df.head(2)
########################################################################################
# However, to access the full native capability of pyrolite, we'd need to rename
# these columns to use things like :func:`~pyrolite.geochem.pyrochem.convert_chemistry`:
#
units = {  # keep a copy of the units, we can use these to map back later
    c: c[c.find("_") + 1 :] if "_" in c else None for c in df.columns
}
df = df.rename(
    columns={c: c.replace("_wt%", "").replace("_ppm", "") for c in df.columns}
)
df.head(2)
########################################################################################
# We could then perform our chemistry conversion, rename our columns to include
# units, and optionally export to e.g. CSV:
#
converted_wt_pct = df.pyrochem.convert_chemistry(
    to=df.pyrochem.list_oxides + ["NiO", "TiO2"]
)
converted_wt_pct.head(2)
########################################################################################
# Here we rename the columns before we export them, just so we know explicitly
# what the units are:
converted_wt_pct = converted_wt_pct.rename(
    columns={c: c + "_wt%" for c in converted_wt_pct.pyrochem.list_oxides}
)
converted_wt_pct.head(2)
########################################################################################
converted_wt_pct.to_csv("converted_wt_pct.csv")
