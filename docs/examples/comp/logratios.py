import numpy as np
import pandas as pd
from pyrolite.comp.codata import ilr, inverse_ilr, close

np.random.seed(82)


def random_compositional_trend(m1, m2, c1, c2, resolution=20, size=1000):
    # generate means intermediate between m1 and m2
    mv = np.vstack([ilr(close(m1)).reshape(1, -1), ilr(close(m2)).reshape(1, -1)])
    ms = np.apply_along_axis(lambda x: np.linspace(*x, resolution), 0, mv)
    # generate covariance matricies intermediate between c1 and c2
    cv = np.vstack([c1.reshape(1, -1), c2.reshape(1, -1)])
    cs = np.apply_along_axis(lambda x: np.linspace(*x, resolution), 0, cv)
    cs = cs.reshape(cs.shape[0], *c1.shape)
    # generate samples from each
    samples = np.vstack(
        [
            np.random.multivariate_normal(m.flatten(), cs[ix], size=size // resolution)
            for ix, m in enumerate(ms)
        ]
    )
    # combine together.
    return inverse_ilr(samples)


m1, m2 = np.array([[0.3, 0.1, 2.1]]), np.array([[0.5, 2.5, 0.05]])
c1, c2 = np.eye(2) / 100, np.eye(2) / 100

trend = pd.DataFrame(
    random_compositional_trend(m1, m2, c1, c2, resolution=100, size=5000)
)
ax = trend.pyroplot.density(mode="density", bins=100)
ax.tax.scatter(
    inverse_ilr(np.nanmean(ilr(trend.values), axis=0)[np.newaxis, :]) * 100,
    marker="D",
    color="k",
    label="LogMean",
)

ax.tax.scatter(
    close(np.nanmean(trend.values, axis=0))[np.newaxis, :] * 100,
    marker="o",
    color="r",
    label="LogMean",
)
ax.figure
