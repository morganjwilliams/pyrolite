import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot.stem import stem

np.random.seed(82)
# %% Minimal Example -------------------------------------------------------------------
# create some example data
x = np.linspace(0, 10, 10) + np.random.randn(10) / 2.0
y = np.random.rand(10)
df = pd.DataFrame(np.vstack([x, y]).T, columns=['Depth', 'Fe3O4'])
# plot
ax = stem(df.Depth, df.Fe3O4, figsize=(5,3))
# or, alternatively directly from the dataframe:
ax = df.pyroplot.stem(figsize=(5,3))
# %% Save Figure
from pyrolite.util.plot import save_figure
save_figure(ax.figure, save_at="../../source/_static", name="stem_minimal")

# %% Orientation --------------------------------------------------------------------------
ax = df.pyroplot.stem(orientation='vertical', figsize=(3,5))
# the yaxes can then be inverted using:
ax.invert_yaxis()
# and if you'd like the xaxis to be labeled at the top:
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="stem_vertical")
