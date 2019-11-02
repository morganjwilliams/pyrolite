import unittest
from pyrolite.mineral.lattice import strain_coefficient
from pyrolite.geochem.ind import get_ionic_radii


class TestStrainCoefficient(unittest.TestCase):
    def setUp(self):
        self.r0 = get_ionic_radii("Ca", charge=2, coordination=8)  # angstroms
        self.ri = get_ionic_radii("Ca", charge=2, coordination=8)
        self.D0 = 4.1
        self.E = 120 * 10 ** 9  # Pa
        self.Tk = 1200.0  #  kelvin

    def test_default(self):
        for rx in [
            get_ionic_radii("Sr", charge=2, coordination=8),
            get_ionic_radii("Mn", charge=2, coordination=8),
            get_ionic_radii("Mg", charge=2, coordination=8),
            get_ionic_radii("Ba", charge=2, coordination=8),
        ]:
            D_j = self.D0 * strain_coefficient(
                self.ri, rx, r0=self.r0, E=self.E, T=self.Tk
            )
            self.assertTrue(D_j > 0.0)
            self.assertTrue(D_j < self.D0)

    def test_modulus_not_specified(self):
        rx = get_ionic_radii("Sr", charge=2, coordination=8)
        D_j = self.D0 * strain_coefficient(self.ri, rx, r0=self.r0, z=2, T=self.Tk)
        self.assertTrue(D_j > 0.0)
        self.assertTrue(D_j < self.D0)


if __name__ == "__main__":
    unittest.main()
