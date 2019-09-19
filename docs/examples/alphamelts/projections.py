from pathlib import Path
from pyrolite.ext.alphamelts.tables import get_experiments_summary
from pyrolite.ext.alphamelts.plottemplates import table_by_phase

tempdir = Path("./") / "montecarlo"

summary = get_experiments_summary(tempdir / "isobar5kbar1300-800C", kelvin=False)

# %%
import matplotlib.pyplot as plt
import pyrolite.plot
from pyrolite.util.plot import add_colorbar

otpt = summary["MORB_Gale2013-0"]["output"]
cmap = plt.cm.magma
norm = plt.Normalize(
    vmin=otpt.liquidcomp.Temperature.min(), vmax=otpt.liquidcomp.Temperature.max()
)

ax = otpt.liquidcomp.loc[:, ["CaO", "MgO", "Al2O3"]].pyroplot.scatter(
    color=otpt.liquidcomp.Temperature.values, cmap=cmap, figsize=(8, 8)
)
add_colorbar(ax.collections[-1])

# %%
import numpy as np
import pandas as pd
from pyrolite.comp.codata import clr, inverse_clr, alr, inverse_alr

# cpx olivine plag
# CaMgSi2O6 Mg2Si2O4 CaAl2Si2O8
components = pd.DataFrame(
    np.array([[1, 1, 0, 2], [0, 2, 0, 2], [1, 0, 1, 2]]),
    columns=["CaO", "MgO", "Al2O3", "SiO2"],
    index=["cpx", "olivine", "plag"],
)

from pyrolite.geochem import get_cations, simple_oxides


def component_matrix(components=[], phases=[], names=[]):
    """
    Break down a list of phases into oxide components.
    """
    atoms = set()
    # get atoms which are not O
    for p in phases:
        atoms |= set(get_cations(p))
    print(atoms)
    oxides = [simple_oxides(a)[1] for a in atoms]
    return oxides


component_matrix(components=X.columns, phases=["CaAl2Si2O8", "Mg2Si2O4", "CaMgSi2O6"])


from pyrolite.util.synthetic import test_df, random_cov_matrix

X = test_df(
    cols=["cpx", "olivine", "plag"],
    cov=random_cov_matrix(dim=2, sigmas=np.array([0.5, 0.7])),
)
mixed = X @ components.to_weight().renormalise(scale=1)


def unmix_compositions(X, components, bdl=10 * -5):
    x = X.copy()

    c = components.copy()
    c = c.to_weight().renormalise(scale=1)
    # c[c == 0] = bdl; clr

    a, b = c.values.T, x.values.T
    a[np.isnan(a)] = 0.0
    b[np.isnan(b)] = 0.0

    A, res, rank, s = np.linalg.lstsq(a, b, rcond=None)
    A = A.T
    A[A == 0] = np.nan

    return pd.DataFrame(A, index=x.index, columns=c.index)


mixed = otpt.liquidcomp.loc[:, components.columns]
U = unmix_compositions(mixed, components)

ax = U.pyroplot.ternary(c=otpt.liquidcomp.Temperature.values, cmap=cmap)
add_colorbar(ax.collections[-1])

# %%


def cost(norm, compositon, components, order=2):
    """
    y = A * x
    components_inv @ compositon = norm
    """
    # compositional difference
    Q = A.T @ A
    c = -A.T @ y
    _Q = 0.5 * x.T @ Q @ x
    _c = c.T @ x
    return np.nansum(_Q + _c)
    """
    c = np.nansum(
        ((clr(X[np.newaxis, :]) - clr((norm.T[np.newaxis, :] @ components))) ** order)
    )

    return c
    """


x0 = np.ones((X.index.size, components.index.size)) / components.index.size
cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
bounds = np.array([[0.0, None], [0.0, None], [0.0, None]])

x = x0.copy()
for ix, row in enumerate(x0):
    row

    x0 = row[np.newaxis, :].T
    x0
    minout = scipy.optimize.minimize(
        cost,
        x0,
        args=(X.values[ix], components),
        bounds=bounds,
        method="SLSQP",
        constraints=cons,
    )
    x[ix] = minout.x
minout.x
x
# %%
components


"""
import scipy
X[np.isnan(X)] = 0
sol, res, rnk, s  = scipy.linalg.lstsq(X.T, A.T)
norm = pd.DataFrame(sol.T, columns=A.index).renormalise(scale=1)
norm[norm==0] = np.nan
norm.loc[:, :] = inverse_clr(norm.values)
"""
# otpt.phasemass.loc[:, ["olivine_0", "clinopyroxene_0", "feldspar_0"]].pyroplot.ternary()
