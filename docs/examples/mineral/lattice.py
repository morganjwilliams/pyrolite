import numpy as np
import matplotlib.pyplot as plt
from pyrolite.geochem import get_ionic_radii
from pyrolite.geochem.ind import REE
from pyrolite.mineral.lattice import strain_coefficient
# %% config
D_Na = 1.35  # Partition coefficient An-Melt
D_Ca = 4.1  # Partition coefficient An-Melt
Tc = 900  # Temperature, °C
Tk = Tc + 273.15  # Temperature, K
E_2 = 120 * 10 ** 9  # Youngs modulus for 2+ site, Pa
E_3 = 135 * 10 ** 9  # Youngs modulus for 3+ site, Pa
rCa = get_ionic_radii("Ca", charge=2, coordination=8)
rLa = get_ionic_radii("La", charge=3, coordination=8)
# %% 2+ cations
fontsize = 8
fig, ax = plt.subplots(1)

site2labels = ["Na", "Ca", "Eu", "Sr"]
# get the Shannon ionic radii for the elements in the 2+ site
site2radii = [
    get_ionic_radii("Na", charge=1, coordination=8),
    *[get_ionic_radii(el, charge=2, coordination=8) for el in ["Ca", "Eu", "Sr"]],
]
# plot the relative paritioning curve for cations in the 2+ site
site2Ds = D_Ca * np.array([strain_coefficient(rCa, rx, E_2, T=Tk) for rx in site2radii])
ax.scatter(site2radii, site2Ds, color="g", label="$X^{2+}$ Cations")
# create an index of radii, and plot the relative paritioning curve for the site
xs = np.linspace(0.9, 1.3, 200)
curve2Ds = D_Ca * strain_coefficient(rCa, xs, E_2, T=Tk)
ax.plot(xs, curve2Ds, color="0.5", ls="--")
# add the element labels next to the points
for l, r, d in zip(site2labels, site2radii, site2Ds):
    ax.annotate(
        l, xy=(r, d), xycoords="data", ha="left", va="bottom", fontsize=fontsize
    )
# %% Calculate D(La)
D_La = (D_Ca ** 2 / D_Na) * np.exp((529 / Tk) - 3.705)
# %% 3+ cations
site3labels = REE()
# get the Shannon ionic radii for the elements in the 3+ site
site3radii = [get_ionic_radii(x, charge=3, coordination=8) for x in REE()]
site3Ds = D_La * np.array([strain_coefficient(rLa, rx, E_3, T=Tk) for rx in site3radii])
# plot the relative paritioning curve for cations in the 3+ site
ax.scatter(site3radii, site3Ds, color="purple", label="$X^{3+}$ Cations")
# plot the relative paritioning curve for the site
curve3Ds = D_La * strain_coefficient(rLa, xs, E_3, T=Tk)
ax.plot(xs, curve3Ds, color="0.5", ls="--")
# add the element labels next to the points
for l, r, d in zip(site3labels, site3radii, site3Ds):
    ax.annotate(
        l, xy=(r, d), xycoords="data", ha="right", va="bottom", fontsize=fontsize
    )

ax.set_yscale("log")
ax.set_ylabel("$D_X$")
ax.set_xlabel("Radii ($\AA$)")
# %% Effective europium anomaly as a function of fraction of Eu3 / (Eu3 + Eu2)
X_Eu3 = 0.6
# calculate D_Eu3 relative to D_La
D_Eu3 = D_La * strain_coefficient(
    rLa, get_ionic_radii("Eu", charge=3, coordination=8), E_3, T=Tk
)
# calculate D_Eu2 relative to D_Ca
D_Eu2 = D_Ca * strain_coefficient(
    rCa, get_ionic_radii("Eu", charge=2, coordination=8), E_2, T=Tk
)
# calculate the effective parition coefficient
D_Eu = (1 - X_Eu3) * D_Eu2 + X_Eu3 * D_Eu3
# show the effective partition coefficient relative to the 2+ and 3+ endmembers
radii, ds = (
    [get_ionic_radii("Eu", charge=c, coordination=8) for c in [3, 3, 2, 2]],
    [D_Eu3, D_Eu, D_Eu, D_Eu2],
)
ax.plot(
    radii, ds, ls="--", color="0.9", marker="D", label="Effective $D_{Eu}$", zorder=-1
)
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False, facecolor=None)
# %% save figure
from pyrolite.util.plot import save_figure

save_figure(fig, save_at="../../source/_static", name="plag_lattice")
