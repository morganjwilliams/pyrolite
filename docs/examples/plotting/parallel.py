import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
import pyrolite.plot
import pyrolite.data.Aitchison

# %% Data
# let's load up an example dataset from Aitchison
df = pyrolite.data.Aitchison.load_coxite()
comp = [
    i for i in df.columns if i not in ["Depth", "Porosity"]
]  # compositional data variables
# %% Default
ax = df.pyroplot.parallel()
# %% save
from pyrolite.util.plot import save_figure

save_figure(ax.figure, save_at="../../source/_static", name="parallel_default")
# %% Default Rescale
ax = df.pyroplot.parallel(rescale=True)
# %% save
save_figure(ax.figure, save_at="../../source/_static", name="parallel_rescale")
# %% CLR
from pyrolite.util.skl.transform import CLRTransform

cmap = "inferno"
compdata = df.copy()
compdata[comp] = CLRTransform().transform(compdata[comp])
ax = compdata.loc[:, comp].pyroplot.parallel(color_by=compdata.Depth.values, cmap=cmap)

# we can add a meaningful colorbar to indicate one variable also, here Depth
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array(df.Depth)
plt.colorbar(sm)
# %% save
save_figure(ax.figure, save_at="../../source/_static", name="parallel_CLR")
# %% CLR Rescale
ax = compdata.loc[:, comp].pyroplot.parallel(
    rescale=True, color_by=compdata.Depth.values, cmap=cmap
)
plt.colorbar(sm)
# %% save
save_figure(ax.figure, save_at="../../source/_static", name="parallel_CLR_rescale")
