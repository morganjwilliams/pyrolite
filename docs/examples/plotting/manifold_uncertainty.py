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
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].set_title("Mapping - True Classes")
ax[1].set_title("Mapping - Classification & Certainty")
a, tfm, mapped = plot_mapping(data, gs.best_estimator_, ax=ax[1], s=30, init="pca")
ax[0].scatter(*mapped.T, c=__DEFAULT_DISC_COLORMAP__(target), s=30)
# %% Save Figure
from pyrolite.util.plot import save_figure

save_figure(fig, save_at="../../source/_static", name="manifold_uncertainty")
