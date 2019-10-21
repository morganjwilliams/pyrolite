import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot
from pyrolite.plot.tern import ternary

np.random.seed(82)
# %% Minimal Example -------------------------------------------------------------------
# create some example data
oxs = ['SiO2', 'MgO', 'CaO']
ys = np.random.rand(3, len(oxs))
ys = np.exp(ys)
df = pd.DataFrame(data=ys, columns=oxs)
df.loc[:, ['SiO2', 'MgO', 'CaO']]
# plot
ax = ternary(df.loc[:, ['SiO2', 'MgO', 'CaO']].values, color="k")
# or, alternatively directly from the dataframe:
ax = df.loc[:, ['SiO2', 'MgO', 'CaO']].pyroplot.ternary(color="k")
# %% Save Figure
from pyrolite.util.plot import save_figure
save_figure(ax.figure, save_at="../../source/_static", name="tern_minimal")

# %% Specify External Axis ------------------------------------------------------------
# The plotting axis can be specified to use exisiting axes:
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))

df.loc[0, ['SiO2', 'MgO', 'CaO']].pyroplot.ternary(ax=ax[0], color="k")
df.loc[1:, ['SiO2', 'MgO', 'CaO']].pyroplot.ternary(ax=ax[1], color="k")

plt.tight_layout()
# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="tern_dual")
