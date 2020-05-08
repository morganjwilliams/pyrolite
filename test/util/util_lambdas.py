import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from pyrolite.util.lambdas import (
    orthogonal_polynomial_constants,
    evaluate_lambda_poly,
    get_lambda_poly_func,
    plot_lambdas_components
)
from pyrolite.util.synthetic import random_cov_matrix
from pyrolite.geochem.ind import REE, get_ionic_radii


class TestOPConstants(unittest.TestCase):
    """Checks the generation of orthagonal polynomial parameters."""

    def setUp(self):
        elements = [i for i in REE(dropPm=True) if i != "Eu"]  # drop Pm, Eu
        self.xs = np.array(get_ionic_radii(elements, coordination=8, charge=3))
        self.default_degree = 5
        self.expect = [
            (),
            (1.05477,),
            (1.00533, 1.12824),
            (0.99141, 1.06055, 1.14552),
            (0.98482, 1.03052, 1.10441, 1.15343),
        ]

    def test_against_original(self):
        """Check that the constants line up with Hugh's paper."""
        ret = orthogonal_polynomial_constants(self.xs, degree=self.default_degree)
        print(ret, self.expect)
        for out, expect in zip(ret, self.expect):
            with self.subTest(out=out, expect=expect):
                if out:
                    self.assertTrue(
                        np.allclose(
                            np.array(out, dtype="float"),
                            np.array(expect, dtype="float"),
                        )
                    )
                else:
                    self.assertEqual(out, ())  # first one should be empty tuple

    def test_xs(self):
        """Tests operation on different x arrays."""
        for xs in [self.xs, self.xs[1:], self.xs[2:-2]]:
            with self.subTest(xs=xs):
                ret = orthogonal_polynomial_constants(xs, degree=self.default_degree)
                self.assertTrue(not len(ret[0]))  # first item is empty
                self.assertTrue(len(ret) == self.default_degree)

    def test_degree(self):
        """Tests generation of different degree polynomial parameters."""

        max_degree = 4
        expected = orthogonal_polynomial_constants(self.xs, degree=max_degree)
        for degree in range(1, max_degree):
            with self.subTest(degree=degree):
                ret = orthogonal_polynomial_constants(self.xs, degree=degree)
                self.assertTrue(not len(ret[0]))  # first item is empty
                self.assertTrue(len(ret) == degree)
                # the parameter values should be independent of the degree.
                allclose = all(
                    [
                        np.allclose(
                            np.array(expected[idx], dtype=float),
                            np.array(tpl, dtype=float),
                        )
                        for idx, tpl in enumerate(ret)
                    ]
                )
                self.assertTrue(allclose)

    def test_tol(self):
        """
        Tests that the optimization of OP parameters can be achieved
        to different tolerancesself.
        Tolerances don't directly translate, so we expand it slightly for
        the test (by a factor prop. to e**(len(ps)+1)).
        """
        eps = np.finfo(float).eps
        hightol_result = orthogonal_polynomial_constants(
            self.xs, degree=self.default_degree, tol=10 ** -16
        )
        for pow in np.linspace(np.log(eps * 1000.0), -5, 3):
            tol = np.exp(pow)
            with self.subTest(tol=tol):
                ret = orthogonal_polynomial_constants(
                    self.xs, degree=self.default_degree, tol=tol
                )
                self.assertTrue(not len(ret[0]))  # first item is empty
                self.assertTrue(len(ret) == self.default_degree)
                for ix, ps in enumerate(ret):
                    if ps:
                        test_tol = tol * np.exp(len(ps) + 1)
                        a = np.array(list(ps), dtype=float)
                        b = np.array(list(hightol_result[ix]), dtype=float)
                        # print( (abs(a)-abs(b)) / ((abs(a)+abs(b))/2)  - test_tol)
                        self.assertTrue(np.allclose(a, b, atol=test_tol))


class TestGetLambdaPolyFunc(unittest.TestCase):
    """Checks the generation of lambda polynomial functions."""

    def setUp(self):
        self.lambdas = np.array([0.1, 1.0, 10.0, 100.0])
        self.xs = np.linspace(0.9, 1.1, 5)

    def test_noparams(self):
        ret = get_lambda_poly_func(self.lambdas, radii=self.xs)
        self.assertTrue(callable(ret))

    def test_noparams_noxs(self):
        with self.assertRaises(AssertionError):
            ret = get_lambda_poly_func(self.lambdas)
            self.assertTrue(callable(ret))

    def test_function_params(self):
        params = orthogonal_polynomial_constants(self.xs, degree=len(self.lambdas))
        ret = get_lambda_poly_func(self.lambdas, params=params)
        self.assertTrue(callable(ret))

class TestPlotLambdasComponents(unittest.TestCase):

    def setUp(self):
        self.lambdas = np.array([0.1, 1.0, 10.0, 100.0])

    def test_default(self):
        ax = plot_lambdas_components(self.lambdas)
        self.assertIsInstance(ax, matplotlib.axes.Axes)


    def tearDown(self):
        plt.close('all')


if __name__ == "__main__":
    unittest.main()
