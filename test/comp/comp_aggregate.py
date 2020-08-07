import unittest
import numpy as np
from pyrolite.comp.aggregate import *
from pyrolite.util.synthetic import normal_frame

import logging


class TestCompositionalMean(unittest.TestCase):
    """Tests pandas compositional mean operator."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = normal_frame(columns=self.cols)

    def test_1D(self):
        """Checks results on single records."""
        df = pd.DataFrame(self.df.iloc[:, 0].head(1))
        out = compositional_mean(df)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out.values, axis=-1), 1.0))

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        out = compositional_mean(df)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out.values, axis=-1), 1.0))

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        out = compositional_mean(df)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out.values, axis=-1), 1.0))

    @unittest.expectedFailure
    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        df = self.df.copy()
        # Create some nans to imitate contrasting analysis sets
        df.iloc[
            np.random.randint(1, 10, size=2),
            np.random.randint(1, len(self.cols), size=2),
        ] = np.nan
        out = compositional_mean(df)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out.values, axis=-1), 1.0))

    def test_mean(self):
        """Checks whether the mean is accurate."""
        pass


class TestWeightsFromArray(unittest.TestCase):
    """Tests the numpy array-weight generator for weighted averages."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = normal_frame(columns=self.cols)

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        out = weights_from_array(df.values)
        self.assertTrue(out.size == 1)

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        out = weights_from_array(df.values)
        self.assertTrue(out.size == df.index.size)


class TestGetFullColumn(unittest.TestCase):
    """Tests the nan-column checking function for numpy arrays."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = normal_frame(columns=self.cols)
        nans = 10
        self.df.iloc[
            np.random.randint(1, 10, size=nans),
            np.random.randint(1, len(self.cols), size=nans),
        ] = np.nan

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        out = get_full_column(df.values)
        self.assertTrue(out == 0)

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        out = get_full_column(df.values)
        self.assertTrue(out == 0)


class TestNANWeightedMean(unittest.TestCase):
    """Tests numpy weighted NaN-mean operator."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = normal_frame(columns=self.cols)

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        out = nan_weighted_mean(df.values)
        self.assertTrue(np.allclose(out, df.values))

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        out = nan_weighted_mean(df.values)
        self.assertTrue(np.allclose(out, np.mean(df.values, axis=0)))

    def test_multiple_equal_weights(self):
        """Checks results on multiple records with equal weights."""
        df = self.df.copy()
        weights = np.array([1.0 / len(df.index)] * len(df.index))
        out = nan_weighted_mean(df.values, weights=weights)
        self.assertTrue(
            np.allclose(out, np.average(df.values, weights=weights, axis=0))
        )

    def test_multiple_unequal_weights(self):
        """Checks results on multiple records with unequal weights."""
        df = self.df.copy()
        weights = np.random.rand(1, df.index.size).squeeze()
        out = nan_weighted_mean(df.values, weights=weights)
        check = np.average(df.values.T, weights=weights, axis=1)
        self.assertTrue(
            np.allclose(out, np.average(df.values, weights=weights, axis=0))
        )

    def test_multiple_unequal_weights_withnan(self):
        """
        Checks results on multiple records with unequal weights,
        where the data includes some null data.
        """
        df = self.df.copy()
        df.iloc[0, :] = np.nan  # make one record nan
        # Some non-negative weights

        weights = np.random.rand(1, df.index.size).squeeze()
        weights = np.array(weights) / np.nansum(weights)
        out = nan_weighted_mean(df.values, weights=weights)
        check = np.average(df.iloc[1:, :].values, weights=weights[1:], axis=0)
        self.assertTrue(np.allclose(out, check))


class TestNANWeightedCompositionalMean(unittest.TestCase):
    """Tests numpy weighted compositonal NaN-mean operator."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = normal_frame(columns=self.cols)
        self.df = self.df.apply(lambda x: x / np.sum(x), axis="columns")

    def test_single(self):
        """Checks results on single records."""
        # Should not change result, once closure is considered
        df = self.df.head(1).copy()
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = nan_weighted_compositional_mean(df.values, renorm=renorm)
                if renorm:
                    self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.0))
                    self.assertTrue(np.allclose(out, df.values.reshape(out.shape)))

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = nan_weighted_compositional_mean(df.values, renorm=renorm)
                if renorm:
                    self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.0))

    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for this function
        df = self.df.copy()
        # Create some nans to imitate contrasting analysis sets
        df.iloc[
            np.random.randint(1, 10, size=2),
            np.random.randint(1, len(self.cols), size=2),
        ] = np.nan
        out = nan_weighted_compositional_mean(df.values)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.0))

    def test_mean(self):
        """Checks whether the mean is accurate."""
        pass


class TestCrossRatios(unittest.TestCase):
    """Tests pandas cross ratios utility."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.d = len(self.cols)
        self.n = 10
        self.df = normal_frame(columns=self.cols, size=self.n)

    def test_single(self):
        """Checks results on single record."""
        df = self.df.head(1).copy()
        n = df.index.size
        out = cross_ratios(df)
        self.assertTrue(np.isfinite(out).any())
        self.assertTrue((out[np.isfinite(out)] > 0).all())
        self.assertTrue(out.shape == (n, self.d, self.d))

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        n = df.index.size
        out = cross_ratios(df)
        self.assertTrue(np.isfinite(out).any())
        self.assertTrue((out[np.isfinite(out)] > 0).all())
        self.assertTrue(out.shape == (n, self.d, self.d))

    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        df = self.df.copy()
        n = df.index.size
        # Create some nans to imitate contrasting analysis sets
        df.iloc[
            np.random.randint(1, self.n, size=2), np.random.randint(1, self.d, size=2)
        ] = np.nan
        out = cross_ratios(df)
        self.assertTrue(np.isfinite(out).any())
        self.assertTrue((out[np.isfinite(out)] > 0).all())
        self.assertTrue(out.shape == (n, self.d, self.d))


class TestNPCrossRatios(unittest.TestCase):
    """Tests numpy cross ratios utility."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.d = len(self.cols)
        self.n = 10
        self.df = normal_frame(columns=self.cols, size=self.n)

    def test_single(self):
        """Checks results on single record."""
        df = self.df.head(1).copy()
        n = df.index.size
        arr = df.values
        out = np_cross_ratios(arr)
        self.assertTrue(np.isfinite(out).any())
        self.assertTrue((out[np.isfinite(out)] > 0).all())
        self.assertTrue(out.shape == (n, self.d, self.d))

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        n = df.index.size
        arr = df.values
        out = np_cross_ratios(arr)
        self.assertTrue(np.isfinite(out).any())
        self.assertTrue((out[np.isfinite(out)] > 0).all())
        self.assertTrue(out.shape == (n, self.d, self.d))

    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        df = self.df.copy()
        n = df.index.size
        # Create some nans to imitate contrasting analysis sets
        df.iloc[
            np.random.randint(1, self.n, size=2), np.random.randint(1, self.d, size=2)
        ] = np.nan
        arr = df.values
        out = np_cross_ratios(arr)
        self.assertTrue(np.isfinite(out).any())
        self.assertTrue((out[np.isfinite(out)] > 0).all())
        self.assertTrue(out.shape == (n, self.d, self.d))


class TestStandardiseAggregate(unittest.TestCase):
    """Tests pandas internal standardisation aggregation method."""

    def setUp(self):
        self.mcols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.mdf = pd.DataFrame(
            {k: v for k, v in zip(self.mcols, np.random.rand(len(self.mcols), 10))}
        )
        self.mdf = self.mdf.apply(lambda x: x / np.sum(x), axis="columns")

        self.tcols = ["SiO2", "Ni", "Cr", "Sn"]
        self.tdf = pd.DataFrame(
            {k: v for k, v in zip(self.tcols, np.random.rand(len(self.tcols), 10))}
        )

        self.df = self.mdf.append(self.tdf, ignore_index=True, sort=False)

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = standardise_aggregate(df, renorm=renorm)
                outvals = out.values[~np.isnan(out.values)]
                dfvals = df.values[~np.isnan(df.values)]
                self.assertTrue(np.allclose(outvals, dfvals))

    def test_multiple_with_IS(self):
        """
        Checks results on multiple records with internal standard specifed.
        """
        df = self.mdf.copy()
        fixed_record_idx = 0
        int_std = "SiO2"
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = standardise_aggregate(
                    df,
                    int_std=int_std,
                    renorm=renorm,
                    fixed_record_idx=fixed_record_idx,
                )
                if not renorm:
                    self.assertTrue(
                        np.allclose(
                            out[int_std],
                            df.iloc[fixed_record_idx, df.columns.get_loc(int_std)],
                        )
                    )

    def test_multiple_without_IS(self):
        """
        Checks results on multiple records without internal standard specifed.
        """
        df = self.mdf
        fixed_record_idx = 0
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = standardise_aggregate(
                    df, renorm=renorm, fixed_record_idx=fixed_record_idx
                )
                if not renorm:
                    self.assertTrue(
                        np.isclose(
                            out.values, df.iloc[fixed_record_idx, :].values
                        ).any()
                    )

    def test_contrasting_with_IS(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for records which differ by all-but-one element
        df = self.df
        fixed_record_idx = 0
        int_std = "SiO2"
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = standardise_aggregate(
                    df,
                    int_std=int_std,
                    renorm=renorm,
                    fixed_record_idx=fixed_record_idx,
                )
                if not renorm:
                    self.assertTrue(
                        np.allclose(
                            out[int_std],
                            df.iloc[fixed_record_idx, df.columns.get_loc(int_std)],
                        )
                    )

    def test_contrasting_without_IS(self):
        """
        Checks results on multiple contrasting records
        without internal standard specifed.
        """
        df = self.df
        fixed_record_idx = 0
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = standardise_aggregate(
                    df, renorm=renorm, fixed_record_idx=fixed_record_idx
                )
                if not renorm:
                    self.assertTrue(
                        np.isclose(
                            out.values, df.iloc[fixed_record_idx, :].values
                        ).any()
                    )


if __name__ == "__main__":
    unittest.main()
