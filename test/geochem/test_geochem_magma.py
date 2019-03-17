import unittest
import numpy as np
from pyrolite.geochem.magma import *


class TestFeAt8MgO(unittest.TestCase):
    def setUp(self):
        self.FeOT = 2
        self.MgO = 4

    def test_default(self):
        feat8 = FeAt8MgO(self.FeOT, self.MgO)

    def test_close_to_8(self):
        feat8 = FeAt8MgO(self.FeOT, 8.0)
        self.assertTrue(np.isclose(feat8, self.FeOT))


class TestNaAt8MgO(unittest.TestCase):
    def setUp(self):
        self.Na2O = 2
        self.MgO = 4

    def test_default(self):
        naat8 = FeAt8MgO(self.Na2O, self.MgO)

    def test_close_to_8(self):
        naat8 = FeAt8MgO(self.Na2O, 8.0)
        self.assertTrue(np.isclose(naat8, self.Na2O))


if __name__ == "__main__":
    unittest.main()
