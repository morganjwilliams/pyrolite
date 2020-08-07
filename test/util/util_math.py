import unittest
import pandas as pd
import numpy as np
from pyrolite.util.math import *
from pyrolite.util.synthetic import random_cov_matrix
from sympy import tensorcontraction


class TestAugmentedCovarianceMatrix(unittest.TestCase):
    def setUp(self):
        self.mean = np.random.randn(5)
        self.cov = random_cov_matrix(5)

    def test_augmented_covariance_matrix(self):
        ACM = augmented_covariance_matrix(self.mean, self.cov)


class TestInterpolateLine(unittest.TestCase):
    def setUp(self):
        self.x, self.y = np.linspace(0.0, 10.0, 10), np.random.randn(10)

    def test_default(self):
        # should do no interpoltion
        ix, iy = interpolate_line(self.x, self.y)
        self.assertTrue(isinstance(ix, np.ndarray))
        self.assertTrue(ix.shape == self.x.shape)

    def test_n(self):
        for n in [2, 5]:
            ix, iy = interpolate_line(self.x, self.y, n=n)
            self.assertTrue(isinstance(ix, np.ndarray))
            self.assertTrue(
                iy.shape[-1] == self.y.shape[-1] + (self.y.shape[-1] - 1) * n
            )


class TestIndexesRanges(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(1, 10, 10)

    def test_linspc(self):
        spc = linspc_(self.x.min(), self.x.max())
        self.assertTrue(np.isclose(spc[0], self.x.min()))
        self.assertTrue(np.isclose(spc[-1], self.x.max()))

    def test_logspc(self):
        spc = logspc_(self.x.min(), self.x.max())
        self.assertTrue(np.isclose(spc[0], self.x.min()))
        self.assertTrue(np.isclose(spc[-1], self.x.max()))

    def test_linrng_default(self):
        # should be equivalent to linspace where all above zero
        rng = linrng_(self.x)
        self.assertTrue(isinstance(rng, tuple))
        self.assertTrue(np.isclose(rng[0], self.x.min()))
        self.assertTrue(np.isclose(rng[1], self.x.max()))

    def test_logrng_default(self):
        rng = logrng_(self.x)
        self.assertTrue(isinstance(rng, tuple))
        self.assertTrue(np.isclose(rng[0], self.x.min()))
        self.assertTrue(np.isclose(rng[1], self.x.max()))


class TestGridFromRanges(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(10, 2)

    def test_default(self):
        out = grid_from_ranges(self.x)
        # default bins = 100
        self.assertTrue(out[0].size == 100 ** 2)

    def test_bins(self):
        for bins in [2, 10, 50]:
            out = grid_from_ranges(self.x, bins=bins)


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
    Tests is_numeric function.
    """

    def test_numeric_collection_instances(self):
        for obj in [
            np.array([]),
            pd.Series([], dtype="float32"),
            pd.DataFrame([], dtype="float32"),
        ]:
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

    def normal_seriesies(self):
        vals = pd.Series(self.values, dtype="float64")
        rounded = round_sig(vals, sig=2)
        self.assertTrue(
            np.isclose(
                rounded, self.expect2.reshape(rounded.shape), equal_nan=True
            ).all()
        )

    def test_dataframe(self):
        vals = pd.DataFrame(self.values, dtype="float64")
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
        for vals, unc, expect in [
            (self.values, self.unc, self.unc_expect),
            (self.values[3], self.unc[3], self.unc_expect[3]),
        ]:
            with self.subTest(vals=vals, unc=unc):
                sfs = significant_figures(vals, unc=unc)
                self.assertTrue(np.allclose(sfs, expect, equal_nan=True))

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

    def normal_seriesies(self):
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

    def normal_seriesies(self):
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

    def normal_seriesies(self):
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
        self.X = np.vstack((np.arange(10.0), -np.arange(10.0))).T
        self.X -= np.nanmean(self.X, axis=0)[np.newaxis, :]
        self.target = np.eye(2) + -1.0 * np.eye(2)[::-1, :]

    def test_simple(self):
        """Checks whether non-nan covariances are correct."""
        X = self.X
        out = nancov(X)
        self.assertTrue(np.allclose(out / out[0][0], self.target))

    def test_one_column_partial_nan(self):
        """Checks whether a single column containing NaN is processed."""
        X = self.X
        X[0, 1] = np.nan
        out = nancov(X)
        self.assertTrue(np.allclose(out / out[0][0], self.target))

    def test_all_column_partial_nan(self):
        """Checks whether all columns containing NaNs is processed."""
        X = self.X
        X[0, 1] = np.nan
        X[1, 0] = np.nan
        out = nancov(X)
        self.assertTrue(np.allclose(out / out[0][0], self.target))

    @unittest.expectedFailure
    def test_one_column_all_nan(self):
        """Checks whether a single column all-NaN is processed."""
        X = self.X
        X[:, 1] = np.nan
        for method in ["replace", "rowexclude"]:
            with self.subTest(method=method):
                out = nancov(X, method=method)
                self.assertTrue(np.allclose(out / out[0][0], self.target))

    @unittest.expectedFailure
    def test_all_column_all_nan(self):
        """Checks whether all columns all-NaNs is processed."""
        X = self.X
        X[:, 1] = np.nan
        X[:, 0] = np.nan
        out = nancov(X)
        self.assertTrue(np.allclose(out / out[0][0], self.target))


class TestHelmertBasis(unittest.TestCase):
    """Test the orthogonal basis generator for ILR transformation."""

    def setUp(self):
        self.X = np.ones((10, 3))

    def test_helmert_basis_default(self):
        basis = helmert_basis(D=self.X.shape[0])


class TestSymbolicHelmert(unittest.TestCase):
    def test_default(self):
        for ix in np.arange(2, 10):
            with self.subTest(ix=ix):
                basis = symbolic_helmert_basis(ix)
                sums = tensorcontraction(basis, (1,))
                self.assertTrue(all([i == 0 for i in sums]))

    def test_full(self):
        for ix in np.arange(2, 10):
            with self.subTest(ix=ix):
                basis = symbolic_helmert_basis(ix, full=True)
                sums = tensorcontraction(basis, (1,))  # first row won't be 0
                self.assertTrue(all([i == 0 for i in sums[1:]]))


if __name__ == "__main__":
    unittest.main()
