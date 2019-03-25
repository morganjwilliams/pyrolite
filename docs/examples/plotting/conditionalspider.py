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
    ("plot", "plot", [], dict(color="k", alpha=0.01)),
    ("fill", "fill", [], dict()),
    ("binkde", "binkde", [], dict(resolution=5)),
    (
        "binkde",
        "binkde percentiles specified",
        [],
        dict(percentiles=[0.95, 0.5], resolution=5),
    ),
    ("ckde", "ckde", [], dict(resolution=5)),
    ("kde", "kde", [], dict(resolution=5)),
    ("histogram", "histogram", [], dict(resolution=5)),
]

fig, ax = plt.subplots(
    len(modes), len(ss), sharey=True, figsize=(len(ss) * 2.5, 1.5 * len(modes))
)

for a, (m, name, args, kwargs) in zip(ax, modes):
    a[0].annotate(  # label the axes rows
        "Mode: {}".format(name),
        xy=(0.1, 1.05),
        xycoords=a[0].transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )
for ix, s in enumerate(ss):
    x = np.arange(nindex)
    y = rc["PM_PON"].normalize(data).applymap(np.log)
    y = np.tile(y, nobs).reshape(nobs, nindex)
    y += np.random.normal(0, s / 2, size=(nobs, nindex))  # noise
    y += np.random.normal(0, s, size=(1, nobs)).T  # random pattern offset
    y[:, 3] += 1.0  # significant offset
    df = pd.DataFrame(np.exp(y), columns=components)
    for mix, (m, name, args, kwargs) in enumerate(modes):
        df.pyroplot.spider(indexes=x, mode=m, ax=ax[mix, ix], *args, **kwargs)

plt.tight_layout()
# %% -----
from pyrolite.util.plot import save_figure, save_axes

save_figure(
    fig,
    save_at="../../source/_static",
    name="spider_modes",
)
