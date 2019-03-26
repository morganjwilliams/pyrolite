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
ss = [0.1, 0.2, 0.5]  # sigmas for noise

modes = [
    ("plot", "plot", [], dict(color="k", alpha=0.01)),
    ("fill", "fill", [], dict()),
    ("binkde", "binkde", [], dict(resolution=5)),
    (
        "binkde",
        "binkde contours specified",
        [],
        dict(contours=[0.95, 0.5], resolution=5),
    ),
    ("ckde", "ckde", [], dict(resolution=5)),
    ("kde", "kde", [], dict(resolution=5)),
    ("histogram", "histogram", [], dict(resolution=5)),
]

fig, ax = plt.subplots(
    len(modes), len(ss), sharey=True, figsize=(len(ss) * 3, 2 * len(modes))
)
ax[0, 0].set_ylim((0.1, 100))

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
    df = pd.DataFrame(y, columns=components)
    df["Eu"] += 1.0  # significant offset
    df = df.applymap(np.exp)
    for mix, (m, name, args, kwargs) in enumerate(modes):
        df.pyroplot.spider(
            indexes=x,
            mode=m,
            ax=ax[mix, ix],
            cmap="viridis",
            vmin=0.05,
            *args,
            **kwargs
        )

plt.tight_layout()
# %% save figure
from pyrolite.util.plot import save_figure, save_axes

save_figure(fig, save_at="../../source/_static", name="spider_modes")
