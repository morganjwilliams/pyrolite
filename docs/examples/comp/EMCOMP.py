import numpy as np
import matplotlib.pyplot as plt
from pyrolite.comp.impute import EMCOMP
from pyrolite.util.synthetic import random_composition

np.random.seed(82)

sample_data = random_composition(100, 5, propnan=0.1, missing="MNAR")

imputed_data, p0, niter = EMCOMP(
    sample_data, threshold=np.nanpercentile(sample_data, 90, axis=0),
    tol=0.01
)
# %% Plot Data --
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 4))

ax[0].set_title('Original Data')
ax[1].set_title('New Imputed Data')
ax[2].set_title('Imputed Dataset')
fltr = ((np.isfinite(imputed_data).sum(axis=1) - np.isfinite(sample_data).sum(axis=1)) > 0)

ax[0].scatter(*sample_data[:, 2:4].T, marker="D", color="0.5", alpha=0.1)
ax[1].scatter(*imputed_data[fltr, 2:4].T, marker="D", color="r", alpha=0.1)
ax[2].scatter(*imputed_data[:, 2:4].T, marker="D", color="k", alpha=0.1)
for a in ax:
    a.set_xlabel('Component 1')
    a.set_ylabel('Component 2')
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
plt.plot(SD, ϕ, color='0.5')
plt.plot(SD, Φ, color='0.5')
plt.plot(SD, ϕ/Φ, color='0.5') # pdf / cdf
plt.scatter(SD, sigma*ϕ/Φ, color='k',label='D')
plt.legend(frameon=False, facecolor=None)
