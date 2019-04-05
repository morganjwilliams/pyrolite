import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from pyrolite.data.Aitchison import *
from pyrolite.comp.codata import clr, inverse_clr


def logmean(df):
    return pd.Series(
        inverse_clr(np.mean(clr(df.values), axis=0)[np.newaxis, :])[0], index=df.columns
    )


# %% Data
df = load_kongite()

fig, ax = plt.subplots(1)
bins = np.linspace(0, 50, 51) * 2
for column in df:
    df[column].plot.hist(ax=ax, bins=bins, alpha=0.5, label=column)
# %% Simple Means and covariance

# For compostitional data, everything is relative, so we tend to use ratios
# Say you want to know the average ratio between A and B
A_on_B = df["A"] / df["B"]
A_on_B.mean()
# Equally, you could have chosen to calculate the average ratio between B and A
B_on_A = df["B"] / df["A"]
B_on_A.mean()
# You expect these to be invertable, such that A_on_B = 1 / B_on_A; but not so!
A_on_B.mean() / (1 / B_on_A.mean())
# Similarly, the relative variances are different:
np.std(A_on_B) / A_on_B.mean()
np.std(B_on_A) / B_on_A.mean()
# This improves when using logratios in place of simple ratios, prior to exponentiating means
logA_on_B = (df["A"] / df["B"]).apply(np.log)
logB_on_A = (df["B"] / df["A"]).apply(np.log)
# The logratios are invertible:
np.isclose(np.exp(logA_on_B.mean()), 1 / np.exp(logB_on_A.mean()))
# The logratios also have the same variance:
np.isclose(
    (np.std(logA_on_B) / logA_on_B.mean()) ** 2,
    (np.std(logB_on_A) / logB_on_A.mean()) ** 2,
)

# %% Higher Dimensional Visualisation of Mean
# This issue of accuracy/validity of means is also seen in higher dimensions:
df = load_kongite()

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flat

for columns, a in zip(itertools.combinations(["A", "B", "C", "D"], 3), ax):
    columns = list(columns)
    df.loc[:, columns].pyroplot.ternary(ax=a, color="k", label=df.name, no_ticks=True)
    df.mean().loc[columns].pyroplot.ternary(
        ax=a, color="red", label="Arithmetic Mean", no_ticks=True
    )
    print(logmean(df.loc[:, columns]))

    logmean(df.loc[:, columns]).pyroplot.ternary(
        ax=a, s=30, color="green", label="Geometric Mean", axlabels=True, no_ticks=True
    )
    a.legend(frameon=False, facecolor=None, loc=(0.8, 0.5))
