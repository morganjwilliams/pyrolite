import numpy as np
import matplotlib.pyplot as plt
from pyrolite.comp.impute import EMCOMP, random_composition_missing

np.random.seed(82)

sample_data = random_composition_missing(size=1000, propnan=0.05, MAR=False)
imputed_data, p0, niter = EMCOMP(sample_data, threshold= np.nanmin(sample_data, axis=0))
# %% Plot Data --
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,4))

ax[0].scatter(*data[:, :2].T, marker='D', color='0.5', alpha=0.1)
ax[1].scatter(*imputed_data[:, :2].T, marker='D', color='0.5', alpha=0.1)
# %% Save Figure --
from pyrolite.util.plot import save_figure
save_figure(fig, save_at="../../source/_static", name="EMCOMP_comparison")
