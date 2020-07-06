import unittest
import pandas as pd
import numpy as np
from pyrolite.plot.density.ternary import ternary_heatmap
from pyrolite.util.skl.transform import ILRTransform, ALRTransform
from pyrolite.comp.codata import ILR, ALR, inverse_ILR, inverse_ALR


class TestTernaryHeatmap(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100, 3)

    def test_default(self):
        out = ternary_heatmap(self.data)
        self.assertTrue(isinstance(out, tuple))
        coords, H, data = out
        self.assertTrue(coords[0].shape == coords[1].shape)
        # zi could have more or less bins depending on mode..

    def test_histogram(self):
        out = ternary_heatmap(self.data, mode="histogram")
        xe, ye, zi = out
        coords, H, data = out
        self.assertTrue(coords[0].shape == coords[1].shape)

    def test_density(self):
        out = ternary_heatmap(self.data, mode="density")
        coords, H, data = out
        self.assertTrue(coords[0].shape == coords[1].shape)

    def test_transform(self):
        for tfm, itfm in [
            (ALR, inverse_ALR),
            (ILR, inverse_ILR),
            (ILRTransform, None),
            (ALRTransform, None),
        ]:
            with self.subTest(tfm=tfm, itfm=itfm):
                out = ternary_heatmap(self.data, transform=tfm, inverse_transform=itfm)
                coords, H, data = out

    @unittest.expectedFailure
    def test_need_inverse_transform(self):
        for tfm, itfm in [(ALR, None), (ILR, None)]:
            with self.subTest(tfm=tfm, itfm=itfm):
                out = ternary_heatmap(self.data, transform=tfm, inverse_transform=itfm)


if __name__ == "__main__":
    unittest.main()
