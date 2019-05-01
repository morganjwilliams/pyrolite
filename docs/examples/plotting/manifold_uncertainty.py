import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from pyrolite.util.skl.pipeline import SVC_pipeline
from pyrolite.util.skl.vis import plot_mapping
from pyrolite.util.plot import __DEFAULT_DISC_COLORMAP__

np.random.seed(82)
# %% Import sklearn digits dataset ---------------------
wine = sklearn.datasets.load_wine()
data, target = wine["data"], wine["target"]
# %% Discard Dimensionality --------------------------------------
# data = data[:, np.random.random(data.shape[1]) > 0.4]  # randomly remove fraction of dimensionality
# %% SVC Pipeline # example of machine learning pipline which is probabilistic across classes
svc = SVC_pipeline(probability=True)
gs = svc.fit(data, target)
# %% Plot
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

a, tfm, mapped = plot_mapping(
    data, gs.best_estimator_, ax=ax[1], s=50, init="pca"
)
ax[0].scatter(*mapped.T, c=__DEFAULT_DISC_COLORMAP__(gs.predict(data)), s=50)
# %% Save Figure
ax[0].set_title("Predicted Classes")
ax[1].set_title("With Relative Certainty")
from pyrolite.util.plot import save_figure
save_figure(fig, save_at="../../source/_static", name="manifold_uncertainty")

# %% figure no axes
ax[0].annotate('a)', xy=(0.05, 0.9),xycoords =ax[0].transAxes, fontsize=20)
ax[1].annotate('b)', xy=(0.05, 0.9),xycoords =ax[1].transAxes, fontsize=20)
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
save_figure(fig, save_at="../../source/_static", name="manifold_uncertainty_noticks")
