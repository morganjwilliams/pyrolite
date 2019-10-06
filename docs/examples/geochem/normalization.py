import pandas as pd
import matplotlib.pyplot as plt
import pyrolite.plot
from pyrolite.geochem.ind import REE
from pyrolite.geochem.norm import get_reference_composition, all_reference_compositions
# %% getref
chondrite = get_reference_composition("Chondrite_PON")
# %% set units
CI = chondrite.set_units("ppm")
# %% REE plot
fig, ax = plt.subplots(1)

for name, ref in list(all_reference_compositions().items())[::2]:
    if name != "Chondrite_PON":
        ref.set_units("ppm")
        ref.comp.pyrochem.REE.pyrochem.normalize_to(
            CI, units="ppm", convert_first=False
        ).pyroplot.REE(unity_line=True, ax=ax, label=name)

ax.set_ylabel("X/X$_{Chondrite}$")
ax.legend(
    frameon=False, facecolor=None, loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=2
)
# %% save_figure
from pyrolite.util.plot import save_figure
save_figure(ax.figure, name="REEvChondrite", save_at="../../source/_static")
