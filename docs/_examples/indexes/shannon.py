import matplotlib.pyplot as plt
from pyrolite.geochem.ind import get_ionic_radii, REE

REE_radii = get_ionic_radii(REE(), coordination=8, charge=3)

fig, ax = plt.subplots(1)
ax.plot(REE_radii, c='k', marker='D')
ax.set_xticks(range(len(REE())))
ax.set_xticklabels(REE())
ax.set_ylabel('Ionic Radius ($\AA$)')
ax.set_title('Rare Earth Element Shannon Ionic Radii')

# %% End -------------------------------------------------------------------------------
from pyrolite.util.plot import save_figure

save_figure(fig, save_at="../../source/_static", name="ShannonRadii")
