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


class TestNaNCov(unittest.TestCase):
    """Tests the numpy nan covariance matrix utility."""

    def setUp(self):
        self.X = np.random.rand(1000, 10)

    def test_simple(self):
        """Checks whether non-nan covariances are correct."""
        X = np.vstack((np.arange(10), -np.arange(10))).T
        out = nancov(X)
        target = np.eye(2) + -1. * np.eye(2)[::-1, :]
        self.assertTrue(np.allclose(out/out[0][0], target))

    def test_replace_method(self):
        """Checks whether the replacement method works."""
        pass

    def test_rowexclude_method(self):
        """Checks whether the traditional row-exclude method works."""
        pass

    def test_one_column_partial_nan(self):
        """Checks whether a single column containing NaN is processed."""
        pass

    def test_all_column_partial_nan(self):
        """Checks whether all columns containing NaNs is processed."""
        pass

    def test_one_column_all_nan(self):
        """Checks whether a single column all-NaN is processed."""
        pass

    def test_all_column_all_nan(self):
        """Checks whether all columns all-NaNs is processed."""
        pass


class TestOrthagonalBasis(unittest.TestCase):
    """Test the orthagonal basis generator for ILR transformation."""

    def test_orthagonal_basis(self):
        """Checks orthagonality of the transformation basis."""
        pass


class TestOPConstants(unittest.TestCase):
    """Checks the generation of orthagonal polynomial parameters."""

    def setUp(self):
        self.xs = np.array(get_radii(REE()))
        self.default_degree = 4

    def test_xs(self):
        """Tests operation on different x arrays."""
        for xs in [self.xs, self.xs[1:], self.xs[2:-2]]:
            with self.subTest(xs=xs):
                ret = OP_constants(xs, degree=self.default_degree)
                self.assertTrue(not len(ret[0])) # first item is empty
                self.assertTrue(len(ret) == self.default_degree)

    def test_degree(self):
        """Tests generation of different degree polynomial parameters."""

        max_degree = 4
        expected = OP_constants(self.xs, degree=max_degree)
        for degree in range(1, max_degree):
            with self.subTest(degree=degree):
                ret = OP_constants(self.xs, degree=degree)
                self.assertTrue(not len(ret[0])) # first item is empty
                self.assertTrue(len(ret) == degree)
                # the parameter values should be independent of the degree.
                allclose = all([np.allclose(np.array(expected[idx], dtype=float),
                                            np.array(tpl, dtype=float))
                                for idx, tpl in enumerate(ret)])
                self.assertTrue(allclose)

    def test_tol(self):
        """
        Tests that the optimization of OP parameters can be achieved
        to different tolerancesself.
        Tolerances don't directly translate, so we expand it slightly for
        the test (by a factor prop. to e**(len(ps)+1)).
        """
        eps = np.finfo(float).eps
        hightol_result = OP_constants(self.xs,
                                      degree=self.default_degree,
                                      tol=10**-16)
        for pow in np.linspace(np.log(eps*1000.), -5, 3):
            tol = np.exp(pow)
            with self.subTest(tol=tol):
                ret = OP_constants(self.xs,
                                   degree=self.default_degree,
                                   tol=tol)
                self.assertTrue(not len(ret[0])) # first item is empty
                self.assertTrue(len(ret) == self.default_degree)
                for ix, ps in enumerate(ret):
                    if ps:
                        test_tol = tol * np.exp(len(ps)+1)
                        a = np.array(list(ps), dtype=float)
                        b = np.array(list(hightol_result[ix]), dtype=float)
                        #print( (abs(a)-abs(b)) / ((abs(a)+abs(b))/2)  - test_tol)
                        self.assertTrue(np.allclose(a, b, atol=test_tol))


class TestLambdaPolyFunc(unittest.TestCase):
    """Checks the generation of lambda polynomial functions."""

    def setUp(self):
        self.lambdas = np.array([0.1, 1., 10., 100.])
        self.xs = np.linspace(0.9, 1.1, 5)

    def test_function_generation(self):
        ret = lambda_poly_func(self.lambdas, self.xs)
        self.assertTrue(callable(ret))


if __name__ == '__main__':
    unittest.main()
