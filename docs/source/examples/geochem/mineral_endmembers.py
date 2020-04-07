"""
Mineral Endmember Decomposition
=================================

A common task when working with mineral chemistry data is to take measured compositions
and decompose these into relative proportions of mineral endmember compositions.
pyrolite includes some utilities to achieve this and a limited mineral database
for looking up endmember compositions. This part of the package is being actively
developed, so expect expansions and improvements soon.
"""
import pandas as pd
import numpy as np
from pyrolite.mineral.mindb import get_mineral
from pyrolite.mineral.normative import endmember_decompose

########################################################################################
# First we'll start with a composition of an unknown olivine:
#
comp = pd.Series({"MgO": 42.06, "SiO2": 39.19, "FeO": 18.75})
########################################################################################
# We can break this down into olivine endmebmers using the
# :func:`~pyrolite.mineral.transform.endmember_decompose` function:
#
ed = endmember_decompose(
    pd.DataFrame(comp).T, endmembers="olivine", ord=1, molecular=True
)
ed
########################################################################################
# Equally, if you knew the likely endmembers beforehand, you could specify a list of
# endmembers:
#
ed = endmember_decompose(
    pd.DataFrame(comp).T, endmembers=["forsterite", "fayalite"], ord=1, molecular=True
)
ed
########################################################################################
# We can check this by recombining the components with these proportions. We can first
# lookup the compositions for our endmembers:
#
em = pd.DataFrame([get_mineral("forsterite"), get_mineral("fayalite")])
em.loc[:, ~(em == 0).all(axis=0)]  # columns not full of zeros
########################################################################################
# First we have to convert these element-based compositions to oxide-based compositions:
#
emvalues = (
    em.loc[:, ["Mg", "Si", "Fe"]]
    .pyrochem.to_molecular()
    .fillna(0)
    .pyrochem.convert_chemistry(to=["MgO", "SiO2", "FeO"], molecular=True)
    .fillna(0)
    .pyrocomp.renormalise(scale=1)
)
emvalues
########################################################################################
# These can now be used with our endmember proportions to regenerate a composition:
#
recombined = pd.DataFrame(ed.values.flatten() @ emvalues).T.pyrochem.to_weight()
recombined
########################################################################################
# To make sure these compositions are within 0.01 percent:
#
assert np.allclose(recombined.values, comp.values, rtol=10 ** -4)
