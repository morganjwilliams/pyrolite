import unittest
import pandas as pd
import numpy as np
from pyrolite.util.math import *
from pyrolite.geochem import REE, get_ionic_radii


class TestIndexsRanges(unittest.TestCase):

    def setUp(self):
        pass

class TestIsClose(unittest.TestCase):
    def test_non_nan(self):
        self.assertTrue(isclose(1.0, 1.0))
        self.assertTrue(isclose(0.0, 0.0))

        self.assertTrue(isclose(np.array([1.0]), np.array([1.0])))
        self.assertTrue(isclose(np.array([0.0]), np.array([0.0])))

    def test_nan(self):
        self.assertTrue(isclose(np.nan, np.nan))
        self.assertTrue(isclose(np.array([np.nan]), np.array([np.nan])))


class TestIsNumeric(unittest.TestCase):
    """
    Tests round_sig function.
    round_sig(x, sig=2)
    """

    def test_numeric_collection_instances(self):
        for obj in [np.array([]), pd.Series([]), pd.DataFrame([])]:
            with self.subTest(obj=obj):
                self.assertTrue(is_numeric(obj))

    def test_numeric_collection_classes(self):
        for obj in [np.ndarray, pd.Series, pd.DataFrame]:
            with self.subTest(obj=obj):
                self.assertTrue(is_numeric(obj))

    def test_number_instances(self):
        for obj in [0, 1, 1.0, 10.0, np.nan, np.inf]:
            with self.subTest(obj=obj):
                self.assertTrue(is_numeric(obj))

    def test_number_classes(self):
        for obj in [np.float, np.int, np.bool, float, int, bool]:
            with self.subTest(obj=obj):
                self.assertTrue(is_numeric(obj))

    def test_bool(self):
        for obj in [True, False]:
            with self.subTest(obj=obj):
                self.assertTrue(is_numeric(obj))

    def test_non_numeric_collection_instances(self):
        for obj in [list(), dict(), set()]:
            with self.subTest(obj=obj):
                self.assertFalse(is_numeric(obj))

    def test_non_numeric_collection_classes(self):
        for obj in [list, dict, set]:
            with self.subTest(obj=obj):
                self.assertFalse(is_numeric(obj))


class TestRoundSig(unittest.TestCase):
    """
    Tests round_sig function.
    round_sig(x, sig=2)
    """

    def setUp(self):
        self.values = [0, 1, 1.1, 2.111, 1232.01, 100000.00, 10000.0001, np.nan, np.inf]
        self.expect2 = np.array([0, 1, 1.1, 2.1, 1200, 100000, 10000, np.nan, np.inf])

    def test_sigs(self):
        vals = np.array(self.values)
        for sig in range(5):
            with self.subTest(sig=sig):
                rounded = round_sig(vals, sig=sig)
                self.assertTrue((significant_figures(rounded) <= sig).all())

    def test_list(self):
        vals = list(self.values)
        rounded = round_sig(vals, sig=2)
        self.assertTrue(
            np.isclose(
                rounded, self.expect2.reshape(rounded.shape), equal_nan=True
            ).all()
        )

    def test_array(self):
        vals = np.array(self.values)
        rounded = round_sig(vals, sig=2)
        self.assertTrue(
            np.isclose(
                rounded, self.expect2.reshape(rounded.shape), equal_nan=True
            ).all()
        )

    def test_series(self):
        vals = pd.Series(self.values)
        rounded = round_sig(vals, sig=2)
        self.assertTrue(
            np.isclose(
                rounded, self.expect2.reshape(rounded.shape), equal_nan=True
            ).all()
        )

    def test_dataframe(self):
        vals = pd.DataFrame(self.values)
        rounded = round_sig(vals, sig=2)
        self.assertTrue(
            np.isclose(
                rounded, self.expect2.reshape(rounded.shape), equal_nan=True
            ).all()
        )


class TestSignificantFigures(unittest.TestCase):
    """
    Tests significant_figures function.
    significant_figures(n, unc=None, max_sf=20)
    """

    def setUp(self):
        self.values = [0, 1, 1.1, 2.111, 1232.01, 100000.00, 10000.0001, np.nan, np.inf]
        self.unc = [0, 0.1, 0.5, 1.0, 10, 1, 100, np.nan, np.inf]
        self.expect = np.array([0, 1, 2, 4, 6, 1, 9, 0, 0])
        self.unc_expect = np.array([0, 2, 2, 1, 3, 6, 3, 0, 0])

    def test_unc(self):
        sfs = significant_figures(self.values, unc=self.unc)
        self.assertTrue(np.isclose(sfs, self.unc_expect, equal_nan=True).all())

    def test_max_sf(self):
        for max_sf in range(1, 10):
            with self.subTest(max_sf=max_sf):
                vals = list(self.values)
                sfigs = significant_figures(vals, max_sf=max_sf)
                close = np.isclose(
                    sfigs, self.expect.reshape(sfigs.shape), equal_nan=True
                )
                # where the number of sig figures is below the max
                # should be as expected
                self.assertTrue(close[self.expect < max_sf].all())

    def test_numbers(self):
        for ix, value in enumerate(self.values):
            with self.subTest(value=value, ix=ix):
                sf = significant_figures(value)
                expect = self.expect[ix]
                self.assertTrue(np.isclose(sf, expect, equal_nan=True))

    def test_list(self):
        vals = list(self.values)
        sfigs = significant_figures(vals)
        self.assertTrue(
            np.isclose(sfigs, self.expect.reshape(sfigs.shape), equal_nan=True).all()
        )

    def test_array(self):
        vals = np.array(self.values)
        sfigs = significant_figures(vals)
        self.assertTrue(
            np.isclose(sfigs, self.expect.reshape(sfigs.shape), equal_nan=True).all()
        )

    def test_series(self):
        vals = pd.Series(self.values)
        sfigs = significant_figures(vals)
        self.assertTrue(
            np.isclose(sfigs, self.expect.reshape(sfigs.shape), equal_nan=True).all()
        )

    def test_dataframe(self):
        vals = pd.DataFrame(self.values)
        sfigs = significant_figures(vals)
        self.assertTrue(
            np.isclose(sfigs, self.expect.reshape(sfigs.shape), equal_nan=True).all()
        )


class TestMostPrecise(unittest.TestCase):
    """
    Tests most_precise function.
    most_precise(array_like)
    """

    def setUp(self):
        self.values = [0, 1, 1.1, 2.111, 1232.01, 100000.00, 10000.0001, np.nan, np.inf]
        self.expect = 10000.0001

    def test_list(self):
        vals = list(self.values)
        mp = most_precise(vals)
        self.assertEqual(mp, self.expect)

    def test_array(self):
        vals = np.array(self.values)
        mp = most_precise(vals)
        self.assertEqual(mp, self.expect)

    def test_series(self):
        vals = pd.Series(self.values)
        mp = most_precise(vals)
        self.assertEqual(mp, self.expect)

    def test_dataframe(self):
        vals = pd.DataFrame(data=np.vstack([self.values] * 2))
        mp = most_precise(vals)
        self.assertTrue((mp == self.expect).all().all())


class TestEqualWithinSignificance(unittest.TestCase):
    """
    Tests equal_within_significance function.
    equal_within_significance(arr, equal_nan=False)
    """

    def setUp(self):
        self.equal_values = [0.1, 0.11, 0.113, 0.14]
        self.unequal_values = [1.1, 0.11, 0.113, 0.14]
        self.twoDequal = [self.equal_values, self.equal_values]
        self.twoDunequal = [self.unequal_values, self.equal_values]

    def test_lists(self):
        eq = list(self.equal_values)
        neq = list(self.unequal_values)
        self.assertTrue(equal_within_significance(eq))
        self.assertFalse(equal_within_significance(neq))

    def test_array(self):
        eq = np.array(self.equal_values)
        neq = np.array(self.unequal_values)
        self.assertTrue(equal_within_significance(eq))
        self.assertFalse(equal_within_significance(neq))

    def test_series(self):
        eq = pd.Series(self.equal_values)
        neq = pd.Series(self.unequal_values)
        self.assertTrue(equal_within_significance(eq))
        self.assertFalse(equal_within_significance(neq))

    def test_dataframe(self):
        eq = pd.DataFrame(data=np.array(self.twoDequal).T)
        neq = pd.DataFrame(data=np.array(self.twoDunequal).T)
        self.assertTrue(equal_within_significance(eq).all())
        self.assertFalse(equal_within_significance(neq).all())
        # print(equal_within_significance(neq))


class TestSignifyDigit(unittest.TestCase):
    """
    Tests convert to significant digits function.
    signify_digit(n, unc=None, leeway=0, low_filter=True)
    """

    def test_int(self):
        """Checks operations on inf values."""
        for digit in [0] + [10 ** n for n in range(5)]:
            with self.subTest(digit=digit):
                self.assertEqual(signify_digit(digit), digit)

    def test_inf(self):
        """Checks operations on inf values."""
        digit = np.inf
        self.assertTrue(np.isnan(signify_digit(digit)))

    def test_nan(self):
        """Checks operations on nan values."""
        digit = np.nan
        self.assertTrue(np.isnan(signify_digit(digit)))


class TestOnFinite(unittest.TestCase):
    """Tests on_finite array operation wrapper."""

    def test_inf(self):
        """Checks operations on inf values."""
        arr = np.array([0.0, 1.0, np.inf, -np.inf])
        for f in [np.min, np.max, np.mean]:
            with self.subTest(f=f):
                result = on_finite(arr, f)
                self.assertTrue(np.isclose(result, f(arr[:2])))

    def test_nan(self):
        """Checks operations on nan values."""
        arr = np.array([0.0, 1.0, np.nan, np.nan])

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
        target = np.eye(2) + -1.0 * np.eye(2)[::-1, :]
        self.assertTrue(np.allclose(out / out[0][0], target))

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


class TestOrthogonalBasis(unittest.TestCase):
    """Test the orthogonal basis generator for ILR transformation."""

    def setUp(self):
        self.X = np.ones((10, 3))

    def test_orthogonal_basis_from_array(self):
        basis = orthogonal_basis_from_array(self.X)

    def test_orthogonal_basis_default(self):
        basis = orthogonal_basis_default(self.X.shape[0])


class TestOPConstants(unittest.TestCase):
    """Checks the generation of orthagonal polynomial parameters."""

    def setUp(self):
        self.xs = np.array(get_ionic_radii(REE(), coordination=8, charge=3))
        self.default_degree = 4

    def test_xs(self):
        """Tests operation on different x arrays."""
        for xs in [self.xs, self.xs[1:], self.xs[2:-2]]:
            with self.subTest(xs=xs):
                ret = OP_constants(xs, degree=self.default_degree)
                self.assertTrue(not len(ret[0]))  # first item is empty
                self.assertTrue(len(ret) == self.default_degree)

    def test_degree(self):
        """Tests generation of different degree polynomial parameters."""

        max_degree = 4
        expected = OP_constants(self.xs, degree=max_degree)
        for degree in range(1, max_degree):
            with self.subTest(degree=degree):
                ret = OP_constants(self.xs, degree=degree)
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
        hightol_result = OP_constants(
            self.xs, degree=self.default_degree, tol=10 ** -16
        )
        for pow in np.linspace(np.log(eps * 1000.0), -5, 3):
            tol = np.exp(pow)
            with self.subTest(tol=tol):
                ret = OP_constants(self.xs, degree=self.default_degree, tol=tol)
                self.assertTrue(not len(ret[0]))  # first item is empty
                self.assertTrue(len(ret) == self.default_degree)
                for ix, ps in enumerate(ret):
                    if ps:
                        test_tol = tol * np.exp(len(ps) + 1)
                        a = np.array(list(ps), dtype=float)
                        b = np.array(list(hightol_result[ix]), dtype=float)
                        # print( (abs(a)-abs(b)) / ((abs(a)+abs(b))/2)  - test_tol)
                        self.assertTrue(np.allclose(a, b, atol=test_tol))


class TestLambdaPolyFunc(unittest.TestCase):
    """Checks the generation of lambda polynomial functions."""

    def setUp(self):
        self.lambdas = np.array([0.1, 1.0, 10.0, 100.0])
        self.xs = np.linspace(0.9, 1.1, 5)

    def test_function_generation(self):
        ret = lambda_poly_func(self.lambdas, self.xs)
        self.assertTrue(callable(ret))


if __name__ == "__main__":
    unittest.main()
