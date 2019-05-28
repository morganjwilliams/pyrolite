"""
Utilities for working with the alphaMELTS
executable and associated tabular data. Note that these are
currently experimental and not affiliated with alphaMELTS.
See the `alphaMELTS site <https://magmasource.caltech.edu/alphamelts/>`__ for more info
[#ref_1]_  [#ref_2]_  [#ref_3]_  [#ref_4]_ [#ref_5]_ [#ref_6]_ [#ref_7]_.

Todo
-----
     * As it is developed, also make use of python-melts.
     * MeltsBatch object
     * Develop functions for automation over a grid (of P, T, H2O, fO2, X)
     * Have an option to aggregate summary data, and options to discard experiment data
     * Expansion of documentation

References
-----------
.. [#ref_1] Ghiorso M. S. and Sack R. O. (1995). Chemical mass transfer in magmatic
        processes IV. A revised and internally consistent thermodynamic model for the
        interpolation and extrapolation of liquid-solid equilibria in magmatic systems
        at elevated temperatures and pressures. Contributions to Mineralogy and
        Petrology 119, 197–212.
        doi: {ghiorso1995}
.. [#ref_2] Ghiorso M. S., Hirschmann M. M., Reiners P. W. and Kress V. C. (2002).
        The pMELTS: A revision of MELTS for improved calculation of phase relations
        and major element partitioning related to partial melting of the mantle to
        3 GPa. Geochemistry, Geophysics, Geosystems 3, 1–35.
        doi: {ghiorso2002}
.. [#ref_3] Asimow P. D., Dixon J. E. and Langmuir C. H. (2004).
        A hydrous melting and fractionation model for mid-ocean ridge basalts:
        Application to the Mid-Atlantic Ridge near the Azores. Geochemistry,
        Geophysics, Geosystems 5.
        doi: {asimow2004}
.. [#ref_4] Smith P. M. and Asimow P. D. (2005).
        Adiabat_1ph: A new public front-end to the MELTS, pMELTS, and pHMELTS models.
        Geochemistry, Geophysics, Geosystems 6.
        doi: {smith2005}
.. [#ref_5] Thompson R. N., Riches A. J. V., Antoshechkina P. M., Pearson D. G.,
        Nowell G. M., Ottley C. J., Dickin A. P., Hards V. L., Nguno A.-K. and
        Niku-Paavola V. (2007). Origin of CFB Magmatism: Multi-tiered Intracrustal
        Picrite–Rhyolite Magmatic Plumbing at Spitzkoppe, Western Namibia, during
        Early Cretaceous Etendeka Magmatism. J Petrology 48, 1119–1154.
        doi: {thompson2007}
.. [#ref_6] Antoshechkina P. M., Asimow P. D., Hauri E. H. and Luffi P. I. (2010).
        Effect of water on mantle melting and magma differentiation, as modeled using
        Adiabat_1ph 3.0. AGU Fall Meeting Abstracts 53, V53C-2264.
.. [#ref_7] Antoshechkina P. M. and Asimow P. D. (2010). Adiabat_1ph 3.0 and the MAGMA
        website: educational and research tools for studying the petrology and
        geochemistry of plate margins. AGU Fall Meeting Abstracts 41, ED41B-0644.
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

from ...util.meta import sphinx_doi_link

__doc__ = __doc__.format(
    ghiorso1995=sphinx_doi_link("10.1007/BF00307281"),
    ghiorso2002=sphinx_doi_link("10.1029/2001GC000217"),
    asimow2004=sphinx_doi_link("10.1029/2003GC000568"),
    smith2005=sphinx_doi_link("10.1029/2004GC000816"),
    thompson2007=sphinx_doi_link("10.1093/petrology/egm012"),
)
__doc__ = str(__doc__).replace("ref", __name__)
from .download import *
from .meltsfile import *
from .parse import *
from .tables import *
from .util import *
from .web import *
from .env import *
from .automation import *
