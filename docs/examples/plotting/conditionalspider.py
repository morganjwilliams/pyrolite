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
ss = [0.05, 0.1, 0.2, 0.5]

fig, ax = plt.subplots(3, len(ss), sharex=True, sharey=True, figsize=(len(ss) * 2, 5))

for ix, s in enumerate(ss):
    x = np.arange(nindex)
    y = np.log(rc["PM_PON"].normalize(data).values.astype(np.float).reshape(1, nindex))
    y = np.tile(y, nobs).reshape(nobs, nindex)
    y += np.random.normal(0, s / 2, size=(nobs, nindex))  # noise
    y += np.random.normal(0, s, size=(1, nobs)).T  # random pattern offset
    y[:, 3] += 1.0  # significant offset
    df = pd.DataFrame(np.exp(y), columns=components)
    df.pyroplot.spider(
        indexes=x, c="k", alpha=max(0.01, s / np.log(nobs)), mode="plot", ax=ax[0, ix]
    )
    df.pyroplot.spider(indexes=x, mode="histogram", ax=ax[1, ix], resolution=5)
    df.pyroplot.spider(
        indexes=x, mode="histogram", ax=ax[2, ix], percentiles=[0.95, 0.5], resolution=5
    )
plt.tight_layout()
