import pandas as pd
import numpy as np
from pyrolite.plot import ternaryplot, densityplot
from pyrolite.compositions import inv_alr
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib

import pint
ureg = pint.UnitRegistry()


def ABC_to_tern_xy(ABC):
    (A, B, C) = ABC
    T = A+B+C
    A_n, B_n, C_n = np.divide(A, T), np.divide(B, T), np.divide(C, T)
    xdata = 100.*((C_n/np.sin(np.pi/3)+A_n/np.tan(np.pi/3.))*np.sin(np.pi/3.))
    ydata = 100.*(2./(3.**0.5))*A_n*np.sin(np.pi/3.)
    return xdata, ydata


def tern_heatmapcoords(data, scale=10, bins=10):
    x, y = ABC_to_tern_xy(data)
    xydata = np.vstack((x, y))
    k = gaussian_kde(xydata)

    tridata = dict()
    step = scale // bins
    for i in np.arange(0, scale+1, step):
        for j in np.arange(0, scale+1-i, step):
            datacoord = i, j
            #datacoord = i+0.5*step, j+0.5*step
            tridata[(i, j)] = np.float(k(np.vstack(datacoord)))

    return tridata


mean = np.array([0,0])
cov = np.array([[0.1,-0.05],[-0.05,0.1]])

df = pd.DataFrame(inv_alr(np.random.multivariate_normal(mean, cov, 10000)))
df = df / df.sum(axis=1)

df.plot.scatter(x=1, y=2)
#ternaryplot(df)
#densityplot(df, bins=100, scale=100)


figsize=6
ax=None
ax = ax or plt.subplots(1, figsize=(figsize, figsize* 3**0.5 * 0.5))[1]
scale = 50.
nbins = 50
colorbar=False
empty_df = pd.DataFrame(columns=df.columns)
components = df.columns.values
data = df.loc[:, components].values
heatmapdata = tern_heatmapcoords(data.T, scale=nbins, bins=nbins)
tax = ternaryplot(empty_df, ax=ax, components=components, scale=scale)
ax = tax.ax
mode='density'
if mode == 'hexbin':
    style = 'hexagonal'
else:
    style = 'triangular'
#tax.heatmap(heatmapdata, scale=scale, style=style, colorbar=colorbar)
ax.figure
eps = np.finfo(np.float).eps
for ((x, y), c) in heatmapdata.items():
    c = np.log(c) - np.log(eps)
    z = scale - x - y
    #print(x, y, z)
    tax.scatter(np.array([[z, y, x]]), c=[c], cmap='viridis')

ax.figure
