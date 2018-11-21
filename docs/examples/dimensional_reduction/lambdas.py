import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pyrolite.geochem import REE, get_radii
from pyrolite.plot import REE_radii_plot
from pyrolite.util.math import lambdas, lambda_poly_func, OP_constants

# %% Generate Some Example Data --------------------------------------------------------
no_analyses = 1000

data_ree = [i for i in REE() if not i in ["Pm"]]
data_radii = np.array(get_radii(data_ree))
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
        lnY[ix] += np.random.randn(1) * 0.6

Y = np.exp(lnY)

ax = REE_radii_plot()
ax.plot(data_radii.T, Y.T, marker="D", alpha=0.01, c="0.5", markerfacecolor="k")
# %% Plot Data -------------------------------------------------------------------------

from pyrolite.util.plot import save_figure

save_figure(ax.figure, save_at="../../source/_static", name="LambdaData")

# %% Reduce to Orthagonal Polynomials --------------------------------------------------
lambda_degree = 4

exclude = ["Ce", "Eu", "Pm"]
if exclude:
    subset_Y = Y[:, [i not in exclude for i in data_ree]]
    subset_ree = [i for i in REE() if not i in exclude]
    subset_radii = np.array(get_radii(subset_ree))

params = OP_constants(subset_radii, degree=lambda_degree)

ls = np.apply_along_axis(
    lambda x: lambdas(x, subset_radii, params=params, degree=4), 1, np.log(subset_Y)
)
# %% Plot the Results ------------------------------------------------------------------
fig, ax = plt.subplots(1, lambda_degree - 1, figsize=(9, 3))
ax_labels = [chr(955) + "$_{}$".format(str(d)) for d in range(lambda_degree)]

for ix in range(lambda_degree - 1):
    ax[ix].scatter(ls[:, ix], ls[:, ix + 1], alpha=0.1, c="k")
    ax[ix].set_xlabel(ax_labels[ix])
    ax[ix].set_ylabel(ax_labels[ix + 1])

plt.tight_layout()
fig.suptitle("Lambdas for Dimensional Reduction", y=1.05)
# %% End -------------------------------------------------------------------------------
from pyrolite.util.plot import save_figure

save_figure(fig, save_at="../../source/_static", name="Lambdas")
