import unittest

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyrolite.geochem.ind import REE, get_ionic_radii
from pyrolite.geochem.norm import get_reference_composition
from pyrolite.util.lambdas import calc_lambdas
from pyrolite.util.lambdas.eval import get_lambda_poly_function, lambda_poly
from pyrolite.util.lambdas.params import orthogonal_polynomial_constants
from pyrolite.util.synthetic import random_cov_matrix


class TestOPConstants(unittest.TestCase):
    """Checks the generation of orthogonal polynomial parameters."""

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
            self.xs, degree=self.default_degree, tol=10**-16
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
                        self.assertTrue(np.allclose(a, b, atol=test_tol))


class TestGetLambdaPolyFunc(unittest.TestCase):
    """Checks the generation of lambda polynomial functions."""

    def setUp(self):
        self.lambdas = np.array([0.1, 1.0, 10.0, 100.0])
        self.xs = np.linspace(0.9, 1.1, 5)

    def test_noparams(self):
        ret = get_lambda_poly_function(self.lambdas, radii=self.xs)
        self.assertTrue(callable(ret))

    def test_noparams_noxs(self):
        with self.assertRaises(AssertionError):
            ret = get_lambda_poly_function(self.lambdas)
            self.assertTrue(callable(ret))

    def test_function_params(self):
        params = orthogonal_polynomial_constants(self.xs, degree=len(self.lambdas))
        ret = get_lambda_poly_function(self.lambdas, params=params)
        self.assertTrue(callable(ret))


class TestCalcLambdas(unittest.TestCase):
    def setUp(self):
        self.C = get_reference_composition("PM_PON")
        self.C.set_units("ppm")
        els = [i for i in REE() if not i == "Pm"]
        vals = self.C[els]
        self.df = pd.DataFrame({k: v for (k, v) in zip(els, vals)}, index=[0])

        self.df = self.df.pyrochem.normalize_to("Chondrite_PON", units="ppm")
        self.df.loc[1, :] = self.df.loc[0, :]
        self.default_degree = 3

    def test_exclude(self):
        """
        Tests the ability to generate lambdas from different element sets.
        """
        for exclude in [["Pm"], ["Pm", "Eu"]]:
            with self.subTest(exclude=exclude):
                ret = calc_lambdas(self.df, exclude=exclude, degree=self.default_degree)
                self.assertTrue(ret.columns.size == self.default_degree)

    def test_degree(self):
        """
        Tests the ability to generate lambdas of different degree.
        """
        for degree in range(1, 3):
            with self.subTest(degree=degree):
                ret = calc_lambdas(self.df, degree=degree)
                self.assertTrue(ret.columns.size == degree)

    def test_algorithm(self):
        """
        Check that both algorithms work as expected.
        """
        for alg in ["ONeill", "opt"]:
            with self.subTest(alg=alg):
                ret = calc_lambdas(self.df, algorithm=alg, degree=self.default_degree)
                self.assertTrue(ret.columns.size == self.default_degree)

    def test_anomalies(self):
        anomalies = ["Eu", "Ce"]
        for alg in ["ONeill", "opt"]:
            for add_uncertainties in [True, False]:
                for add_X2 in [True, False]:
                    with self.subTest(
                        alg=alg, add_uncertainties=add_uncertainties, add_X2=add_X2
                    ):
                        ret = calc_lambdas(
                            self.df,
                            algorithm=alg,
                            degree=self.default_degree,
                            add_uncertainties=add_uncertainties,
                            add_X2=add_X2,
                            anomalies=anomalies,
                        )
                        self.assertTrue(
                            ret.columns.size
                            == (
                                self.default_degree * [1, 2][add_uncertainties]
                                + int(add_X2)
                                + len(anomalies)
                            )
                        )
                        self.assertTrue(
                            all(
                                [
                                    "{}/{}*".format(a, a) in ret.columns
                                    for a in anomalies
                                ]
                            )
                        )

    def test_tetrads(self):
        """
        Check that tetrads can be calculated using the optimization algorithm,
        and that where the O'Neill algorithm is specified it will default back to
        the optimization algorithm instead (i.e. still produce a result).
        """
        for alg in ["ONeill", "opt"]:
            with self.subTest(alg=alg):
                ret = calc_lambdas(
                    self.df, algorithm=alg, degree=self.default_degree, fit_tetrads=True
                )
                self.assertTrue(ret.columns.size == self.default_degree + 4)

    def test_opt_fit_method(self):
        for fit_method in ["opt", "lin"]:
            with self.subTest(fit_method=fit_method):
                ret = calc_lambdas(
                    self.df,
                    algorithm="opt",
                    degree=self.default_degree,
                    fit_tetrads=True,
                    fit_method=fit_method,
                )
                self.assertTrue(ret.columns.size == self.default_degree + 4)

    def test_opt_add_uncertainties(self):
        for fit_method in ["opt", "lin"]:
            with self.subTest(fit_method=fit_method):
                ret = calc_lambdas(
                    self.df,
                    algorithm="opt",
                    degree=self.default_degree,
                    fit_tetrads=True,
                    fit_method=fit_method,
                    add_uncertainties=True,
                )
                # should have lambdas, tetrads and uncertainties for both
                self.assertTrue(ret.columns.size == (self.default_degree + 4) * 2)

    def test_params(self):
        # the first three all have the same result - defaulting to ONeill 2016
        # the two following use a full REE set for the OP basis function definition
        # the last illustrates that custom parameters could be used
        degree = self.default_degree
        for params in [
            None,
            "ONeill2016",
            "O'Neill (2016)",
            "full",
            "Full",
            orthogonal_polynomial_constants(
                get_ionic_radii(REE(), charge=3, coordination=8),  # xs
                degree=self.default_degree,
            ),
        ]:
            with self.subTest(params=params):
                ret = calc_lambdas(self.df, params=params, degree=degree)
                self.assertTrue(ret.columns.size == self.default_degree)

    def test_all_nan(self):
        """
        Check that no values are returned for an empty row.
        """
        df = self.df.copy()
        df.loc[1, :] = np.nan
        ret = calc_lambdas(df)
        # all of the second row should be nan
        self.assertTrue((~np.isfinite(ret.iloc[1, :])).values.flatten().all())


if __name__ == "__main__":
    unittest.main()
