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
from ..util.log import Handle

logger = Handle(__name__)


@update_docstring_references
def strain_coefficient(ri, rx, r0=None, E=None, T=298.15, z=None, **kwargs):
    r"""
    Calculate the lattice strain associated with an ionic substitution [#ref_1]_ [#ref_2]_.

    Parameters
    -----------
    ri : :class:`float`
        Ionic radius to calculate strain relative to, in angstroms (Å).
    rj : :class:`float`
        Ionic radius to calculate strain for, in angstroms (Å).
    r0 : :class:`float`, :code:`None`
        Fictive ideal ionic radii for the site. The value for :code:`ri` will be used in its place
        if none is given, and a warning issued.
    E : :class:`float`, :code:`None`
        Young's modulus (stiffness) for the site, in pascals (Pa). Will be estimated using
        :func:`youngs_modulus_approximation` if none is given.
    T : :class:`float`
        Temperature, in Kelvin (K).
    z : :class:`int`
        Optional specification of cationic valence, for calcuation of approximate
        Young's modulus using :func:`youngs_modulus_approximation`,
        where the modulus is not specified.

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
    if r0 is None:
        logger.warn("Use fictive ideal cation radii where possible.")
        r0 = ri
    ri, rx, r0 = ri / 10 ** 10, rx / 10 ** 10, r0 / 10 ** 10  # convert to meters
    E = E or youngs_modulus_approximation(z, r0)  # use E if defined, else try to calc
    coeff = 4.0 * np.pi * E

    rterm = (r0 / 2.0) * (ri - rx) ** 2 - (1.0 / 3) * (ri - rx) ** 3
    return np.exp(coeff * rterm * (-n / (8.314 * T)))


@update_docstring_references
def youngs_modulus_approximation(z, r):
    r"""
    Young's modulus approximation for cationic sites in silicates
    and oxides [#ref_1]_ [#ref_2]_ [#ref_3]_.

    Parameters
    ----------
    z : :class:`integer`
        Cationic valence.
    r : :class:`float`
        Ionic radius of the cation (Å).

    Returns
    --------
    E : :class:`float`
        Young's modulus for the cationic site, in Pascals (Pa)
    Notes
    ------
    The bulk modulus :math:`K` for an an ionic crystal is esimated using [#ref_1]_:

    .. math::

        K = \frac{A Z_a Z_c e^2 (n-1)}{9 d_0 V_0}

    Where :math:`A` is the Madelung constant, :math:`Z_c` and :math:`Z_a` are the
    anion and cation valences, :math:`e` is the charge on the electron,
    :math:`n` is the Born power law coefficent, and :math:`d_0` is the cation-anion
    distance [#ref_1]_. Using the Shannon ionic radius for oxygen (1.38 Å),
    this is approximated for cations coordinated by oxygen in silicates and oxides
    using the following relationship [#ref_2]_:

    .. math::

        K = 750 Z_c d^{-3}

    Where :math:`d` is the cation-anion distance (Å), :math:`Z_c` is the
    cationic valence (and :math:`K` is in GPa). The Young's modulus :math:`E` is
    then calculated through the relationship [#ref_3]_:

    .. math::

        E = 3 K (1 - 2 \sigma)

    Where :math:`\sigma` is Poisson's ratio, which in the case of minerals can be
    approimxated by 0.25 [#ref_3]_, and hence:

    .. math::

        \begin{align}
            E &\approx 1.5 K\\
            E &\approx 1025 Z_c d^{-3}
        \end{align}

    Todo
    -----

        * Add links to docstring

    """
    assert (z is not None) and (
        r is not None
    ), "Need charge and radii to approximate Young's Modulus"
    d = r + 1.38
    E = 1.5 * 750 * (z / d ** 3) * 10 ** 9
    return E


__doc__ = __doc__.format(
    brice1975=sphinx_doi_link("10.1016/0022-0248(75)90241-9"),
    blundy1994=sphinx_doi_link("10.1038/372452a0"),
    anderson1970=sphinx_doi_link("10.1029/JB075i017p03494"),
    hazen1979=sphinx_doi_link("10.1029/JB084iB12p06723"),
    wood2014=sphinx_doi_link("10.1016/B978-0-08-095975-7.00209-6"),
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
    blundy1994=sphinx_doi_link("10.1038/372452a0"),
)
sc_ref = sc_ref.replace(
    "ref", strain_coefficient.__module__ + "." + strain_coefficient.__name__
)
strain_coefficient.__doc__ = strain_coefficient.__doc__ + sc_ref

bm_ref = r"""References
    -----------
        .. [#ref_1] Anderson, D.L., Anderson, O.L., 1970.
                Brief report: The bulk modulus-volume relationship for oxides.
                Journal of Geophysical Research (1896-1977) 75, 3494–3500.
                doi: {anderson1970}
        .. [#ref_2] Hazen, R.M., Finger, L.W., 1979. Bulk modulus—volume relationship for
                cation-anion polyhedra. Journal of Geophysical Research: Solid Earth 84,
                6723–6728.
                doi: {hazen1979}
        .. [#ref_3] Wood, B.J., Blundy, J.D., 2014. Trace Element Partitioning: The Influences of Ionic
                Radius, Cation Charge, Pressure, and Temperature. Treatise on Geochemistry
                (Second Edition) 3, 421–448.
                doi: {wood2014}
    """.format(
    anderson1970=sphinx_doi_link("10.1029/JB075i017p03494"),
    hazen1979=sphinx_doi_link("10.1029/JB084iB12p06723"),
    wood2014=sphinx_doi_link("10.1016/B978-0-08-095975-7.00209-6"),
)
bm_ref = bm_ref.replace(
    "ref",
    youngs_modulus_approximation.__module__
    + "."
    + youngs_modulus_approximation.__name__,
)
youngs_modulus_approximation.__doc__ = youngs_modulus_approximation.__doc__ + bm_ref
