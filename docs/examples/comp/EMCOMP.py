import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrolite.comp.impute import EMCOMP
from pyrolite.util.synthetic import random_composition

np.random.seed(82)

sample_data = random_composition(1000, 4, propnan=0.2, missing="MNAR", missingcols=3)
imputed_data, p0, niter = EMCOMP(
    sample_data, threshold=np.nanpercentile(sample_data, 90, axis=0), tol=0.01
)
imputed_data = pd.DataFrame(imputed_data, columns=["A", "B", "C", "D"])
# %% Plot Data --
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 5))

ax[0].set_title("Original Data")
ax[1].set_title("New Imputed Data")
ax[2].set_title("Imputed Dataset")
fltr = (np.isfinite(sample_data).sum(axis=1)) == sample_data.shape[1]
imputed_data.loc[fltr, ["A", "B", "C"]].pyroplot.ternary(
    marker="D", color="0.5", alpha=0.1, ax=ax[0], no_ticks=True
)
imputed_data.loc[~fltr, ["A", "B", "C"]].pyroplot.ternary(
    marker="D", color="r", alpha=0.1, ax=ax[1], no_ticks=True
)
imputed_data.loc[:, ["A", "B", "C"]].pyroplot.ternary(
    marker="D", color="k", alpha=0.1, ax=ax[2], no_ticks=True
)
# %% Save Figure --
from pyrolite.util.plot import save_figure

save_figure(fig, save_at="../../source/_static", name="EMCOMP_comparison")
# %% --
import scipy.stats

sigma = 0.1
dif = np.random.randn(15)
SD = np.sort(dif / sigma)
ϕ = scipy.stats.norm.pdf(SD, loc=0, scale=1)
Φ = scipy.stats.norm.cdf(SD, loc=0, scale=1)
plt.plot(SD, ϕ, color="0.5")
plt.plot(SD, Φ, color="0.5")
plt.plot(SD, ϕ / Φ, color="0.5")  # pdf / cdf
plt.scatter(SD, sigma * ϕ / Φ, color="k", label="D")
plt.legend(frameon=False, facecolor=None)
