import unittest
import pandas as pd
import numpy as np
from pyrolite.util.math import *
from pyrolite.geochem import REE, get_radii


class TestOnFinite(unittest.TestCase):
    """Tests on_finite array operation wrapper."""

    def test_inf(self):
        """Checks operations on inf values."""
        arr = np.array([0., 1., np.inf, -np.inf])

        for f in [np.min, np.max, np.mean]:
            with self.subTest(f=f):
                result = on_finite(arr, f)
                self.assertTrue(np.isclose(result, f(arr[:2])))

    def test_nan(self):
        """Checks operations on nan values."""
        arr = np.array([0., 1., np.nan, np.nan])

        for f in [np.min, np.max, np.mean]:
            with self.subTest(f=f):
                result = on_finite(arr, f)
                self.assertTrue(np.isclose(result, f(arr[:2])))


class TestOPConstants(unittest.TestCase):

    def setUp(self):
        self.xs = np.array(get_radii(REE()))
        self.default_degree = 5

    def test_xs(self):
        for xs in [self.xs, self.xs[1:], self.xs[2:-2]]:
            with self.subTest(xs=xs):
                ret = OP_constants(xs, degree=self.default_degree)
                self.assertTrue(not len(ret[0])) # first item is empty
                self.assertTrue(len(ret) == self.default_degree)

    def test_degree(self):
        for degree in range(1, 6):
            with self.subTest(degree=degree):
                ret = OP_constants(self.xs, degree=degree)
                self.assertTrue(not len(ret[0])) # first item is empty
                self.assertTrue(len(ret) == degree)

    def test_tol(self):
        pass

        #self.assertTrue(np.allclose(ret, X, atol=tol)


if __name__ == '__main__':
    unittest.main()
