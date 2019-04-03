import matplotlib.pyplot as plt
import sklearn.datasets
from pyrolite.util.skl.pipeline import SVC_pipeline
from pyrolite.util.skl.vis import plot_mapping
from pyrolite.util.plot import __DEFAULT_DISC_COLORMAP__

# %% Import sklearn digits dataset ---------------------
digits = sklearn.datasets.load_digits()  # 64 dimensions, 1800 samples
data, target = digits["data"], digits["target"]
# %% Discard Dimensionality --------------------------------------
# data = data[:, np.random.random(data.shape[1]) > 0.8]  # remove fraction of dimensionality
# example of machine learning pipline which is probabilistic across classes
# %% SVC Pipeline
svc = SVC_pipeline(probability=True)
gs = svc.fit(data, target)
# %% Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].set_title("Mapping - Classfificaitons")
ax[1].set_title("Mapping - Classification & Certainty")
a, tfm, mapped = plot_mapping(
    data, gs.best_estimator_, ax=ax[1], alpha=0.5, s=10, init="pca"
)
ax[0].scatter(*mapped.T, c=__DEFAULT_DISC_COLORMAP__(target), s=10)
# %% --
from pyrolite.util.plot import save_figure

save_figure(fig, save_at="../../source/_static", name="PlotMapping")
