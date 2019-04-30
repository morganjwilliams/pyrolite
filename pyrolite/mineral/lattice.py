import numpy as np


def strain_coefficient(r0, rx, E=None, T=298.15, **kwargs):
    r"""
    Calculate the lattice strain associated with an ionic substitution.

    Parameters
    -----------
    r0 : :class:`float`
        Ionic radius to calculate strain relative to, in angstroms (\AA).
    rj : :class:`float`
        Ionic radius to calculate strain for, in angstroms (\AA).
    E : :class:`float`
        Young's modulus (stiffness) for the site, in pascals (Pa).
    T : :class:`float`
        Temperature, in Kelvin (K).

    Returns
    --------
    :class:`float`

    Notes
    ------

        The lattice strain model relates changes in paritioning to differences in
        ionic radii for ions of a gieven cationic charge, and for a for a specific site
        (with Young's modulus :math:`E`):

        .. math::

            D_{j^{n+}} = D_{A^{n+}} \cdot e^{\frac{-4\pi E N \Big(\frac{r_{0}}{2}(r_j - r_0)^2 + \frac{1}{3}(r_j - r_0)^3\Big)}{RT}}

        Where :math:`D_{A^{n+}}` is the partition coefficient for the ideal ion A, and
        N is Avagadro's number (:math:`6.023 \times 10^{23} mol^{-1}`).

        The 'segregation coefficient' :math:`K_j` can be expressed as:

        .. math::

            K_j = e^{\frac{-\Delta G}{RT}}

        Where :math:`K_j` can be expressed relative to the non-doped equilibirum
        constant :math:`K_0`:

        .. math::

            \begin{align}
            K_j &= e^{\frac{-\Delta G_0 -\Delta G_{strain}}{RT}}\\
                &= e^{\frac{-\Delta G_0}{RT}} \cdot e^{\frac{-\Delta G_{strain}}{RT}}\\
                &= K_0 \cdot e^{\frac{-\Delta G_{strain}}{RT}}\\
            \end{align}

        The model assumes that the crystal is elastically isotropic.

    References
    ----------
        Brice, J.C., 1975. Some thermodynamic aspects of the growth of strained crystals.
        Journal of Crystal Growth 28, 249–253. https://doi.org/10.1016/0022-0248(75)90241-9

        Blundy, J., Wood, B., 1994. Prediction of crystal–melt partition coefficients from elastic moduli.
        Nature 372, 452. https://doi.org/10.1038/372452a0

        Blundy, J., Wood, B., 2003. Partitioning of trace elements between crystals and melts.
        Earth and Planetary Science Letters 210, 383–397. https://doi.org/10.1016/S0012-821X(03)00129-8

        Wood, B.J., Blundy, J.D., 2014. Trace Element Partitioning: The Influences of Ionic
        Radius, Cation Charge, Pressure, and Temperature. Treatise on Geochemistry
        (Second Edition) 3, 421–448. https://doi.org/10.1016/B978-0-08-095975-7.00209-6


    """
    r0, rx = r0 / 10 ** 10, rx / 10 ** 10  # convert to meters
    coeff = -4 * np.pi * E * 6.023 * 10 ** 23 / (8.314 * T)
    rterm = (r0 / 2) * (rx - r0) ** 2 - (1 / 3) * (rx - r0) ** 3
    return np.exp(coeff * rterm)
