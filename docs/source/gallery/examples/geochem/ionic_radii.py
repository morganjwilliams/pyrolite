"""
Ionic Radii
=============

:mod:`pyrolite` incldues a few sets of reference tables for ionic radii in aangstroms
(Å) from [Shannon1976]_ and [WhittakerMuntus1970]_, each with tables indexed
by element, ionic charge and coordination. The easiset way to access these is via
the :func:`~pyrolite.geochem.ind.get_ionic_radii` function. The function can be used
to get radii for individual elements:
"""
from pyrolite.geochem.ind import get_ionic_radii, REE

Cu_radii = get_ionic_radii("Cu")
print(Cu_radii)
########################################################################################
# Note that this function returned a series of the possible radii, given specific
# charges and coordinations of the Cu ion. If we completely specify these, we'll get
# a single number back:
#
Cu2plus6fold_radii = get_ionic_radii("Cu", coordination=6, charge=2)
print(Cu2plus6fold_radii)
########################################################################################
# You can also pass lists to the function. For example, if you wanted to get the Shannon
# ionic radii of Rare Earth Elements (REE) in eight-fold coordination with a valence of
# +3, you should use the following:
#
shannon_ionic_radii = get_ionic_radii(REE(), coordination=8, charge=3)
print(shannon_ionic_radii)
########################################################################################
# The function defaults to using the Shannon ionic radii consistent with [Pauling1960]_,
# but you can adjust to use the set you like with the `pauling` boolean argument
# (:code:`pauling=False` to use Shannon's 'Crystal Radii') or the `source` argument
# (:code:`source='Whittaker'` to use the [WhittakerMuntus1970]_ dataset):
#
shannon_crystal_radii = get_ionic_radii(REE(), coordination=8, charge=3, pauling=False)
whittaker_ionic_radii = get_ionic_radii(
    REE(), coordination=8, charge=3, source="Whittaker"
)
########################################################################################
# We can see what the differences between these look like across the REE:
#
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)

ax.plot(shannon_ionic_radii, marker="D", label="Shannon Ionic Radii")
ax.plot(shannon_crystal_radii, marker="D", label="Shannon Crystal Radii")
ax.plot(whittaker_ionic_radii, marker="D", label="Whittaker & Muntus\nIonic Radii")
{a: b for (a, b) in zip(REE(), whittaker_ionic_radii)}
ax.set_xticks(range(len(REE())))
ax.set_xticklabels(REE())
ax.set_ylabel("Ionic Radius ($\AA$)")
ax.set_title("Rare Earth Element Ionic Radii")
ax.legend(facecolor=None, frameon=False, bbox_to_anchor=(1, 1))

########################################################################################
# .. seealso::
#
#   Examples:
#    `lambdas: Parameterising REE Profiles <lambdas.html>`__,
#    `REE Radii Plot <../plotting/REE_radii_plot.html>`__
#
#   Functions:
#     :func:`~pyrolite.geochem.ind.get_ionic_radii`,
#     :func:`pyrolite.geochem.ind.REE`,
#     :func:`~pyrolite.geochem.pyrochem.lambda_lnREE`,
#
#
# References
# ----------
# .. [Shannon1976] Shannon RD (1976). Revised effective ionic radii and systematic
#         studies of interatomic distances in halides and chalcogenides.
#         Acta Crystallographica Section A 32:751–767.
#         `doi: 10.1107/S0567739476001551 <https://doi.org/10.1107/S0567739476001551>`__.
# .. [WhittakerMuntus1970] Whittaker, E.J.W., Muntus, R., 1970.
#        Ionic radii for use in geochemistry.
#        Geochimica et Cosmochimica Acta 34, 945–956.
#        `doi: 10.1016/0016-7037(70)90077-3 <https://doi.org/10.1016/0016-7037(70)90077-3>`__.
# .. [Pauling1960] Pauling, L., 1960. The Nature of the Chemical Bond.
#         Cornell University Press, Ithaca, NY.
#
