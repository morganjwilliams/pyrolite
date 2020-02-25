import unittest
import numpy as np
import matplotlib.pyplot as plt

from pyrolite.comp.codata import close
from pyrolite.util.plot.transform import xy_to_ABC, ABC_to_xy


class TestTernaryTransforms(unittest.TestCase):
    def setUp(self):
        self.ABC = np.array(
            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1 / 3, 1 / 3, 1 / 3], [1, 1, 1]]
        )
        self.xy = np.array([[0, 0], [0.5, 1.0], [1, 0], [0.5, 1 / 3], [0.5, 1 / 3]])

    def test_xy_to_ABC(self):
        out = xy_to_ABC(self.xy)
        self.assertTrue(np.allclose(out, close(self.ABC)))

    def test_ABC_to_xy(self):
        out = ABC_to_xy(self.ABC)
        self.assertTrue(np.allclose(out, self.xy))

    def test_tfm_inversion_xyABC(self):
        out = ABC_to_xy(xy_to_ABC(self.xy))
        self.assertTrue(np.allclose(out, self.xy))

    def test_tfm_inversion_ABCxy(self):
        out = xy_to_ABC(ABC_to_xy(self.ABC))
        self.assertTrue(np.allclose(out, close(self.ABC)))

    def test_xy_to_ABC_yscale(self):
        for yscale in [1.0, 2.0, np.sqrt(3) / 2]:
            out = xy_to_ABC(self.xy, yscale=yscale)
            expect = self.ABC.copy()
            # scale is slightly complicated; will leave for now
            # test inverse
            self.assertTrue(np.allclose(ABC_to_xy(out, yscale=yscale), self.xy))

    def test_ABC_to_xy_yscale(self):
        for yscale in [1.0, 2.0, np.sqrt(3) / 2]:
            out = ABC_to_xy(self.ABC, yscale=yscale)
            expect = self.xy.copy()
            expect[:, 1] *= yscale
            # test scale
            self.assertTrue(np.allclose(out, expect))
            # test inverse

            self.assertTrue(np.allclose(xy_to_ABC(out, yscale=yscale), close(self.ABC)))


if __name__ == "__main__":
    unittest.main()
