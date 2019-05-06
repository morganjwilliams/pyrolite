"""
Submodule for calcuating relative ion paritioning based on the lattice strain model [#ref_1]_ [#ref_2]_ [#ref_3]_.

Todo
------

    * Bulk modulus and Youngs modulus approximations [#ref_3]_ [#ref_4]_  [#ref_5]_.

References
------------
    .. [#ref_1] Brice, J.C., 1975. Some thermodynamic aspects of the growth of strained crystals.
            Journal of Crystal Growth 28, 249–253.
            doi: {brice1975}
    .. [#ref_2] Blundy, J., Wood, B., 1994. Prediction of crystal–melt partition coefficients from elastic moduli.
            Nature 372, 452.
            doi: {blundy1994}
    .. [#ref_3] Wood, B.J., Blundy, J.D., 2014. Trace Element Partitioning: The Influences of Ionic
            Radius, Cation Charge, Pressure, and Temperature. Treatise on Geochemistry
            (Second Edition) 3, 421–448.
            doi: {wood2014}
    .. [#ref_4] Anderson, D.L., Anderson, O.L., 1970.
            Brief report: The bulk modulus-volume relationship for oxides.
            Journal of Geophysical Research (1896-1977) 75, 3494–3500.
            doi: {anderson1970}
    .. [#ref_5] Hazen, R.M., Finger, L.W., 1979. Bulk modulus—volume relationship for cation-anion polyhedra.
            Journal of Geophysical Research: Solid Earth 84, 6723–6728.
            doi: {hazen1979}

"""
import numpy as np
from ..util.meta import sphinx_doi_link, update_docstring_references


@update_docstring_references
def strain_coefficient(r0, rx, E=None, T=298.15, **kwargs):
    r"""
    Calculate the lattice strain associated with an ionic substitution [#ref_1]_ [#ref_2]_.

    Parameters
    -----------
    r0 : :class:`float`
        Ionic radius to calculate strain relative to, in angstroms (Å).
    rj : :class:`float`
        Ionic radius to calculate strain for, in angstroms (Å).
    E : :class:`float`
        Young's modulus (stiffness) for the site, in pascals (Pa).
    T : :class:`float`
        Temperature, in Kelvin (K).

    Returns
    --------
    :class:`float`
        The strain coefficent :math:`e^{\frac{-\Delta G_{strain}}{RT}}`.

    Notes
    ------

        The lattice strain model relates changes in paritioning to differences in
        ionic radii for ions of a given cationic charge, and for a for a specific site
        (with Young's modulus :math:`E`). This is calcuated using the work
        done to expand a spherical shell centred on the lattice site,
        which alters the :math:`\Delta G` for the formation of the mineral.
        This can be related to changes in partition coefficients using the following
        [#ref_2]_:

        .. math::

            D_{j^{n+}} = D_{A^{n+}} \cdot e^{\frac{-4\pi E N \Big(\frac{r_{0}}{2}(r_j - r_0)^2 + \frac{1}{3}(r_j - r_0)^3\Big)}{RT}}

        Where :math:`D_{A^{n+}}` is the partition coefficient for the ideal ion A, and
        N is Avagadro's number (6.023e23 atoms/mol). This can also
        be calcuated relative to an 'ideal' fictive ion which has a maximum :math:`D`
        where this data are available. This relationship arises via i) the integration
        to calcuate the strain energy mentioned above
        (:math:`4\pi E (\frac{r_{0}}{2}(r_j - r_0)^2 + \frac{1}{3}(r_j - r_0)^3)`),
        and ii) the assumption that the changes in :math:`\Delta G` occur only to size
        differences, and the difference is additive. The 'segregation coefficient'
        :math:`K_j` can be expressed relative to the non-doped equilibirum constant
        :math:`K_0` [#ref_1]_:

        .. math::

            \begin{align}
            K_j &= e^{\frac{-\Delta G_0 -\Delta G_{strain}}{RT}}\\
                &= e^{\frac{-\Delta G_0}{RT}} \cdot e^{\frac{-\Delta G_{strain}}{RT}}\\
                &= K_0 \cdot e^{\frac{-\Delta G_{strain}}{RT}}\\
            \end{align}

        The model assumes that the crystal is elastically isotropic.

    """
    n = 6.023 * 10 ** 23
    r0, rx = r0 / 10 ** 10, rx / 10 ** 10  # convert to meters
    coeff = 4 * np.pi * E
    rterm = (r0 / 2) * (rx - r0) ** 2 - (1 / 3) * (rx - r0) ** 3
    return np.exp(coeff * rterm * (-n / (8.314 * T)))

__doc__ = __doc__.format(
    brice1975=sphinx_doi_link("10.1016/0022-0248(75)90241-9"),
    blundy1994=sphinx_doi_link("10.1038/372452a0"),
    anderson1970=sphinx_doi_link("10.1029/JB075i017p03494"),
    hazen1979=sphinx_doi_link("10.1029/JB084iB12p06723"),
    wood2014=sphinx_doi_link("10.1016/B978-0-08-095975-7.00209-6")
)
__doc__ = str(__doc__).replace("ref", __name__)

# add references for the lattice strain model, needs to be done here due to nested {}
# in the math blocks
sc_ref = r"""References
    ----------
    .. [#ref_1] Brice, J.C., 1975. Some thermodynamic aspects of the growth of strained crystals.
            Journal of Crystal Growth 28, 249–253.
            doi: {brice1975}
    .. [#ref_2] Blundy, J., Wood, B., 1994. Prediction of crystal–melt partition coefficients from elastic moduli.
            Nature 372, 452.
            doi: {blundy1994}
    """.format(
        brice1975=sphinx_doi_link("10.1016/0022-0248(75)90241-9"),
        blundy1994=sphinx_doi_link("10.1038/372452a0")
)
sc_ref = sc_ref.replace("ref", strain_coefficient.__name__)
strain_coefficient.__doc__ = strain_coefficient.__doc__ + sc_ref
