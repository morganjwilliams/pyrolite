r"""
Submodule for calculation and transformation of mass spectrometry data (particularly
for ion-counting and isotope ratio data). Currently in the early stages of development.

Todo
------
    * Dodson ratios [#ref_1]_
    * Add noise-corrected quasi-unbiased ratios of Coath et al. (2012) [#ref_2]_
    * Deadtime methods [#ref_3]_ [#ref_4]_ [#ref_5]_
    * Synthetic data, file format parsing
    * Capability for dealing with reference materials

        This would be handy for :mod:`pyrolite.geochem` in general, and could be
        implemented in a similar way to how :mod:`pyrolite.geochem.norm` handles
        reference compositions.

    * Stable isotope calculations
    * Simple radiogenic isotope system calculations and plots
    * U-Pb, U-series data reduction and uncertainty propagation


References
-----------
.. [#ref_1] Dodson M. H. (1978) A linear method for second-degree interpolation in
            cyclical data collection. Journal of Physics E:
            Scientific Instruments 11, 296.
            doi: {dodson1978}
.. [#ref_2] Coath C. D., Steele R. C. J. and Lunnon W. F. (2012).
            Statistical bias in isotope ratios. J. Anal. At. Spectrom. 28, 52–58.
            doi: {coath2012}
.. [#ref_3] Tyler B. J. (2014). The accuracy and precision of the advanced Poisson
            dead-time correction and its importance for multivariate analysis of
            high mass resolution ToF-SIMS data.
            Surface and Interface Analysis 46, 581–590.
            doi: {tyler2014}
.. [#ref_4] Takano A., Takenaka H., Ichimaru S. and Nonaka H. (2012). Comparison of a
            new dead-time correction method and conventional models for SIMS analysis.
            Surface and Interface Analysis 44, 1287–1293.
            doi: {takano2012}
.. [#ref_5] Müller J. W. (1991). Generalized dead times. Nuclear Instruments and Methods
            in Physics Research Section A: Accelerators, Spectrometers, Detectors and
            Associated Equipment 301, 543–551.
            doi: {muller1991}
"""
from ...util.meta import sphinx_doi_link

__doc__ = __doc__.format(
    coath2012=sphinx_doi_link("10.1039/C2JA10205F"),
    tyler2014=sphinx_doi_link("10.1002/sia.5543"),
    takano2012=sphinx_doi_link("10.1002/sia.5004"),
    muller1991=sphinx_doi_link("10.1016/0168-9002(91)90021-H"),
    dodson1978=sphinx_doi_link("10.1088/0022-3735/11/4/004"),
)
__doc__ = str(__doc__).replace("ref", __name__)

from ...util.log import Handle

logger = Handle(__name__)

from .count import deadtime_correction
# from .background import *
# from .isobaric import *
