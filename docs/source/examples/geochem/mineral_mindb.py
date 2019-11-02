"""
Mineral Database
====================

pyrolite includes a limited mineral database which is useful for
for looking up endmember compositions. This part of the package is being actively
developed, so expect expansions and improvements soon.
"""
import pandas as pd
from pyrolite.mineral.mindb import (
    list_groups,
    list_minerals,
    list_formulae,
    get_mineral,
    get_mineral_group,
)

pd.set_option("precision", 3)  # smaller outputs
########################################################################################
# From the database, you can get the list of its contents using a few utility
# functions:
list_groups()
########################################################################################
list_minerals()
########################################################################################
list_formulae()
########################################################################################
# You can also directly get the composition of specific minerals by name:
#
get_mineral("forsterite")
########################################################################################
# If you want to get compositions for all minerals within a specific group, you can
# use :func:`~pyrolite.mineral.mindb.get_mineral_group`:
get_mineral_group("olivine")
