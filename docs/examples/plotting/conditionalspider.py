import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrolite.geochem.norm import ReferenceCompositions
from pyrolite.geochem.ind import REE
from pyrolite.plot.spider import spider
import logging

rc = ReferenceCompositions()
rn = rc["EMORB_SM89"]  # emorb composition as a starting point
components = [i for i in rn.data.index if i in REE()]
data = rn[components]["value"]
nindex, nobs = data.index.size, 200
ss = [0.05, 0.1, 0.2, 0.5]  # sigmas for noise

modes = [
    ("plot", [], dict(color="k", alpha=0.01)),
    ("fill", [], dict()),
    ("binkde", [], dict(resolution=5)),
    ("ckde", [], dict(resolution=5)),
    ("kde", [], dict(resolution=5)),
    ("histogram", [], dict(resolution=5)),
    ("binkde", [], dict(percentiles=[0.95, 0.5], resolution=5)),
]

fig, ax = plt.subplots(
    len(modes), len(ss), sharey=True, figsize=(len(ss) * 2.5, 2 * len(modes))
)

for a, (m, args, kwargs) in zip(ax, modes):
    a[0].annotate(  # label the axes rows
        "Mode: {}".format(m),
        xy=(0.1, 0.9),
        xycoords=a[0].transAxes,
        fontsize=10,
        ha="left",
        va="top",
    )
for ix, s in enumerate(ss):
    x = np.arange(nindex)
    y = rc["PM_PON"].normalize(data).applymap(np.log)
    y = np.tile(y, nobs).reshape(nobs, nindex)
    y += np.random.normal(0, s / 2, size=(nobs, nindex))  # noise
    y += np.random.normal(0, s, size=(1, nobs)).T  # random pattern offset
    y[:, 3] += 1.  # significant offset
    df = pd.DataFrame(np.exp(y), columns=components)
    for mix, (m, args, kwargs) in enumerate(modes):
        df.pyroplot.spider(indexes=x, mode=m, ax=ax[mix, ix], *args, **kwargs)
plt.tight_layout()

# %% -----
from pyrolite.util.plot import save_figure, save_axes
for a, (m, args, kwargs) in zip(ax, modes):
    save_axes(
        a,
        save_at="../../source/_static",
        name="spider_mode_{}".format(m),
        pad=(0, -0.3),
    )
