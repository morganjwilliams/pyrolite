import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrolite.geochem.norm import ReferenceCompositions
from pyrolite.geochem.ind import common_elements, REE
from pyrolite.plot.spider import spider
import logging

rc = ReferenceCompositions()
rn = rc["EMORB_SM89"]
components = [i for i in rn.data.index if i in REE()]
data = rn[components]["value"]
nindex, nobs = data.index.size, 1000
ss = [0.05, 0.1, 0.2, 0.5]  # sigmas for noise

modes = [
    ("plot", [], dict(color="k", a=max(0.01, s / np.log(nobs)))),
    ("fill", [], dict()),
    ("binkde", [], dict(resolution=5)),
    ("ckde", [], dict(resolution=5)),
    ("kde", [], dict(resolution=5)),
    ("histogram", [], dict(resolution=5)),
    ("binkde", [], dict(percentiles=[0.95, 0.5], resolution=5)),
]

fig, ax = plt.subplots(
    len(modes),
    len(ss),
    sharex=True,
    sharey=True,
    figsize=(len(ss) * 2.5, 1.5 * len(modes)),
)

for ix, s in enumerate(ss):
    x = np.arange(nindex)
    y = np.log(rc["PM_PON"].normalize(data).values.astype(np.float).reshape(1, nindex))
    y = np.tile(y, nobs).reshape(nobs, nindex)
    y += np.random.normal(0, s / 2, size=(nobs, nindex))  # noise
    y += np.random.normal(0, s, size=(1, nobs)).T  # random pattern offset
    y[:, 3] += 1.0  # significant offset
    df = pd.DataFrame(np.exp(y), columns=components)
    for mix, (m, args, kwargs) in enumerate(modes):
        df.pyroplot.spider(indexes=x, mode=m, ax=ax[mix, ix], *args, **kwargs)
plt.tight_layout()
