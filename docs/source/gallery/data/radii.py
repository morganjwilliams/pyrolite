"""
Ionic Radii
==============

:mod:`pyrolite` incldues a few sets of reference tables for ionic radii in aangstroms
(Å) from [Shannon1976]_ and [WhittakerMuntus1970]_, each with tables indexed
by element, ionic charge and coordination. The easiset way to access these is via
the :func:`~pyrolite.geochem.ind.get_ionic_radii` function. The function can be used
to get radii for individual elements, using a :code:`source` keyword argument to swap
between the datasets:
"""
import pandas as pd
import matplotlib.pyplot as plt
from pyrolite.geochem.ind import get_ionic_radii, REE

REE_radii = pd.Series(
    get_ionic_radii(REE(), coordination=8, charge=3, source="Whittaker"), index=REE()
)
REE_radii
########################################################################################
REE_radii.pyroplot.spider(color="k", logy=False)
plt.show()
########################################################################################
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
# .. seealso::
#
#   Examples:
#    `Ionic Radii <../examples/geochem/ionic_radii.html>`__,
#    `lambdas: Parameterising REE Profiles <../examples/geochem/lambdas.html>`__,
#    `REE Radii Plot <../examples/plotting/REE_radii_plot.html>`__
#
#   Functions:
#     :func:`~pyrolite.geochem.ind.get_ionic_radii`,
#     :func:`pyrolite.geochem.ind.REE`,
#     :func:`~pyrolite.geochem.pyrochem.lambda_lnREE`,
#
