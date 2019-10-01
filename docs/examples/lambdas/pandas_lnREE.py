import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pyrolite.plot import pyroplot
from pyrolite.geochem.ind import REE, get_ionic_radii
from pyrolite.geochem.transform import lambda_lnREE
from pyrolite.plot.spider import REE_v_radii
from pyrolite.util.math import lambdas, lambda_poly_func, OP_constants

np.random.seed(82)
# %% Generate Some Example Data --------------------------------------------------------
no_analyses = 1000
data_ree = REE(dropPm=True)
data_radii = np.array(get_ionic_radii(data_ree, charge=3, coordination=8))
data_radii = np.tile(data_radii, (1, no_analyses)).reshape(
    no_analyses, data_radii.shape[0]
)

noise = np.random.randn(*data_radii.shape) * 0.1
constant = -0.1
lin = np.tile(np.linspace(3.0, 0.0, data_radii.shape[1]), (no_analyses, 1))
lin = (lin.T * (1.1 + 0.4 * np.random.rand(data_radii.shape[0]))).T
quad = -1.2 * (data_radii - 1.11) ** 2.0

lnY = noise + constant + lin + quad

for ix, el in enumerate(data_ree):
    if el in ["Ce", "Eu"]:
        lnY[:, ix] += np.random.rand(no_analyses) * 0.6
data_radii
df = pd.DataFrame(np.exp(lnY), columns=data_ree)
ax = df.pyroplot.REE(
    marker="D",
    alpha=0.01,
    c="0.5",
    markerfacecolor="k",
)
# %% Plot Data -------------------------------------------------------------------------
from pyrolite.util.plot import save_figure

save_figure(ax.figure, save_at="../../source/_static", name="PandasLambdaData")
# %% Reduce to Orthogonal Polynomials --------------------------------------------------
ls = lambda_lnREE(df, exclude=["Ce", "Eu", "Pm"], degree=4, norm_to="Chondrite_PON")
# %% Plot the Results ------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax_labels = [chr(955) + "$_{}$".format(str(d)) for d in range(4)]
columns = [chr(955) + str(d) for d in range(4)]

for ix, a in enumerate(ax):
    a.scatter(ls[columns[ix]], ls[columns[ix + 1]], alpha=0.1, c="k")
    a.set_xlabel(ax_labels[ix])
    a.set_ylabel(ax_labels[ix + 1])

plt.tight_layout()
fig.suptitle("Lambdas for Dimensional Reduction", y=1.05)
# %% End -------------------------------------------------------------------------------
from pyrolite.util.plot import save_figure

save_figure(fig, save_at="../../source/_static", name="PandasLambdas")
