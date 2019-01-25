import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot import REE_radii_plot
from pyrolite.geochem.ind import REE, get_radii
# %% Minimal Example -------------------------------------------------------------------
# Where data is not specified, it will return a formatted axis which can be used for
# subsequent plotting:
ax = REE_radii_plot()

# create some example data
ree = REE()
xs = get_radii(ree)
ys = np.linspace(1, 20, len(xs))

ax.plot(xs, ys, marker='D', color='k')
# %% Save Figure
from pyrolite.util.plot import save_figure
save_figure(ax.figure, save_at="../../source/_static", name="REE_radii_plot_minimal")

# %% Generate Some Example Data -------------------------------------------------------
no_analyses = 10

data_ree = [i for i in REE() if not i in ["Pm"]]
data_radii = np.array(get_radii(data_ree))
data_radii = np.tile(data_radii, (1, no_analyses)).reshape(
    no_analyses, data_radii.shape[0]
)

dataframes = []

for i in range(2):
    noise = np.random.randn(*data_radii.shape) * 0.1
    constant = -0.1
    lin = np.tile(np.linspace(3.0, 0.0, data_radii.shape[1]), (no_analyses, 1))
    lin = (lin.T * (1.1 + i/2 * np.random.rand(data_radii.shape[0]))).T
    quad = -1.2 * (data_radii - 1.11) ** 2.0

    lnY = noise + constant + lin + quad

    for ix, el in enumerate(data_ree):
        if el in ["Ce", "Eu"]:
            lnY[:, ix] += np.random.rand(no_analyses) * 0.6

    Y = np.exp(lnY)

    df = pd.DataFrame(Y, columns=data_ree)
    dataframes.append(df)

df1 = dataframes[0]
df2 = dataframes[1]
# %% Data Specified --------------------------------------------------------------------
# Where data is specified, the default plot is a line-based spiderplot:
ax = REE_radii_plot(df1)
# or, alternatively directly from the dataframe:
ax = df1.REE_radii_plot()
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="REE_radii_plot_df")

# %% Fill Plot -------------------------------------------------------------------------
# This behaviour can be modified (see spiderplot docs) to provide filled ranges:
ax = REE_radii_plot(df1, fill=True, plot=False)
# or, alternatively directly from the dataframe:
ax = df1.REE_radii_plot(fill=True, plot=False)
# %% Save Figure
save_figure(ax.figure, save_at="../../source/_static", name="REE_radii_plot_fill")
# %% Specify External Axis ------------------------------------------------------------
# The plotting axis can be specified to use exisiting axes:
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 8))

df1.REE_radii_plot(ax=ax[0])
df2.REE_radii_plot(ax=ax[1], color='k')
plt.tight_layout()

# %% Save Figure
save_figure(fig, save_at="../../source/_static", name="REE_radii_plot_dual")
