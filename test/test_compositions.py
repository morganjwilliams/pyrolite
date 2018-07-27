import unittest
import numpy as np
from pyrolite.compositions import *
import logging
log = logging.getLogger(__name__)


def test_df(cols=['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2'],
            index_length=10):
    return pd.DataFrame({k: v for k,v in zip(cols,
                         np.random.rand(len(cols), index_length))})


class TestClose(unittest.TestCase):
    """Tests array closure operator."""

    def setUp(self):
        self.X1_1R =  np.ones((1)) * 0.2
        self.X1_10R =  np.ones((10, 1)) * 0.2
        self.X10_1R = np.ones((10)) * 0.2
        self.X10_10R = np.ones((10, 10)) * 0.2

    def test_closure_1D(self):
        """Checks that the closure operator works for records of 1 dimension."""
        for X in [self.X1_1R, self.X1_10R]:
            with self.subTest(X=X):
                out = close(X)
                self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.))

    def test_single(self):
        """Checks results on single records."""
        out = close(self.X10_1R)
        self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.))

    def test_multiple(self):
        """Checks results on multiple records."""
        out = close(self.X10_10R)
        self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.))


class TestCompositionalMean(unittest.TestCase):
    """Tests pandas compositional mean operator."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = test_df(cols=self.cols)

    def test_1D(self):
        """Checks results on single records."""
        df = pd.DataFrame(self.df.iloc[:, 0].head(1))
        out = compositional_mean(df)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out.values, axis=-1), 1.))

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        out = compositional_mean(df)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out.values, axis=-1), 1.))

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        out = compositional_mean(df)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out.values, axis=-1), 1.))

    @unittest.expectedFailure
    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        df = self.df.copy()
        # Create some nans to imitate contrasting analysis sets
        df.iloc[np.random.randint(1, 10, size=2),
                np.random.randint(1, len(self.cols), size=2)] = np.nan
        out = compositional_mean(df)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out.values, axis=-1), 1.))

    def test_mean(self):
        """Checks whether the mean is accurate."""
        pass

class TestWeightsFromArray(unittest.TestCase):
    """Tests the numpy array-weight generator for weighted averages."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = test_df(cols=self.cols)

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


class TestGetNonNanColumn(unittest.TestCase):
    """Tests the nan-column checking function for numpy arrays."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = test_df(cols=self.cols)
        nans = 10
        self.df.iloc[np.random.randint(1, 10, size=nans),
                np.random.randint(1, len(self.cols), size=nans)] = np.nan

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        out = get_nonnan_column(df.values)
        self.assertTrue(out == 0)

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        out = get_nonnan_column(df.values)
        self.assertTrue(out == 0)



class TestNANWeightedMean(unittest.TestCase):
    """Tests numpy weighted NaN-mean operator."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = test_df(cols=self.cols)

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
        weights = np.array([1./ len(df.index)] * len(df.index))
        out = nan_weighted_mean(df.values, weights=weights)
        self.assertTrue(np.allclose(out, np.average(df.values,
                                                    weights=weights,
                                                    axis=0))
                                                    )

    def test_multiple_unequal_weights(self):
        """Checks results on multiple records with unequal weights."""
        df = self.df.copy()
        weights = np.random.rand(1, df.index.size).squeeze()
        out = nan_weighted_mean(df.values, weights=weights)
        check = np.average(df.values.T, weights=weights, axis=1)
        self.assertTrue(np.allclose(out, np.average(df.values,
                                                    weights=weights,
                                                    axis=0))
                                                    )

    def test_multiple_unequal_weights_withnan(self):
        """
        Checks results on multiple records with unequal weights,
        where the data includes some null data.
        """
        df = self.df.copy()
        df.iloc[0, :] = np.nan # make one record nan
        # Some non-negative weights

        weights = np.random.rand(1, df.index.size).squeeze()
        weights = np.array(weights)/np.nansum(weights)
        out = nan_weighted_mean(df.values, weights=weights)
        check = np.average(df.iloc[1:, :].values, weights=weights[1:], axis=0)
        self.assertTrue(np.allclose(out, check))



class TestNANWeightedCompositionalMean(unittest.TestCase):
    """Tests numpy weighted compositonal NaN-mean operator."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = test_df(cols=self.cols)
        self.df = self.df.apply(lambda x: x/np.sum(x), axis='columns')

    def test_single(self):
        """Checks results on single records."""
        # Should not change result, once closure is considered
        df = self.df.head(1).copy()
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = nan_weighted_compositional_mean(df.values, renorm=renorm)
                if renorm:
                    self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.))
                    self.assertTrue(np.allclose(out,
                                                df.values.reshape(out.shape)))

    def test_multiple(self):
        """Checks results on multiple records."""
        df = self.df.copy()
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = nan_weighted_compositional_mean(df.values, renorm=renorm)
                if renorm:
                    self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.))

    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for this function
        df = self.df.copy()
        # Create some nans to imitate contrasting analysis sets
        df.iloc[np.random.randint(1, 10, size=2),
                np.random.randint(1, len(self.cols), size=2)] = np.nan
        out = nan_weighted_compositional_mean(df.values)
        # Check closure
        self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.))

    def test_mean(self):
        """Checks whether the mean is accurate."""
        pass


class TestCrossRatios(unittest.TestCase):
    """Tests pandas cross ratios utility."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.d = len(self.cols)
        self.n = 10
        self.df = test_df(cols=self.cols, index_length=self.n)

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
        df.iloc[np.random.randint(1, self.n, size=2),
                np.random.randint(1, self.d, size=2)] = np.nan
        out = cross_ratios(df)
        self.assertTrue(np.isfinite(out).any())
        self.assertTrue((out[np.isfinite(out)] > 0).all())
        self.assertTrue(out.shape == (n, self.d, self.d))


class TestNPCrossRatios(unittest.TestCase):
    """Tests numpy cross ratios utility."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.d = len(self.cols)
        self.n = 10
        self.df = test_df(cols=self.cols, index_length=self.n)

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
        df.iloc[np.random.randint(1, self.n, size=2),
                np.random.randint(1, self.d, size=2)] = np.nan
        arr = df.values
        out = np_cross_ratios(arr)
        self.assertTrue(np.isfinite(out).any())
        self.assertTrue((out[np.isfinite(out)] > 0).all())
        self.assertTrue(out.shape == (n, self.d, self.d))


class TestPDImputeRatios(unittest.TestCase):

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.d = len(self.cols)
        self.n = 10
        self.df = test_df(cols=self.cols, index_length=self.n)

    def test_imputation(self):
        """Checks results on single record."""
        df = self.df.head(1).copy()
        n = df.index.size
        df.iloc[:, np.random.randint(1, self.d, size=1)] = np.nan
        imputed = impute_ratios(df)


class TestNPImputeRatios(unittest.TestCase):

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.d = len(self.cols)
        self.n = 10
        self.df = test_df(cols=self.cols, index_length=self.n)

    def test_imputation(self):
        """Checks results on single record."""
        df = self.df.head(1).copy()
        n = df.index.size
        df.iloc[:, np.random.randint(1, self.d, size=1)] = np.nan
        arr = df.values # single array
        ratios = np_cross_ratios(arr)[0]
        imputed = np_impute_ratios(ratios)


class TestStandardiseAggregate(unittest.TestCase):
    """Tests pandas internal standardisation aggregation method."""

    def setUp(self):
        self.mcols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.mdf = pd.DataFrame({k: v for k,v in zip(self.mcols,
                                np.random.rand(len(self.mcols), 10))})
        self.mdf = self.mdf.apply(lambda x: x/np.sum(x), axis='columns')

        self.tcols = ['SiO2', 'Ni', 'Cr', 'Sn']
        self.tdf = pd.DataFrame({k: v for k,v in zip(self.tcols,
                                np.random.rand(len(self.tcols), 10))})

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
        int_std = 'SiO2'
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = standardise_aggregate(df,
                                            int_std=int_std,
                                            renorm=renorm,
                                            fixed_record_idx=fixed_record_idx)
                if not renorm:
                    self.assertTrue(np.allclose(out[int_std],
                                    df.iloc[fixed_record_idx,
                                            df.columns.get_loc(int_std)])
                                            )

    def test_multiple_without_IS(self):
        """
        Checks results on multiple records without internal standard specifed.
        """
        df = self.mdf
        fixed_record_idx = 0
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = standardise_aggregate(df,
                                            renorm=renorm,
                                            fixed_record_idx=fixed_record_idx)
                if not renorm:
                    self.assertTrue(np.isclose(out.values,
                                    df.iloc[fixed_record_idx, :].values).any()
                                            )

    def test_contrasting_with_IS(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for records which differ by all-but-one element
        df = self.df
        fixed_record_idx = 0
        int_std = 'SiO2'
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = standardise_aggregate(df,
                                            int_std=int_std,
                                            renorm=renorm,
                                            fixed_record_idx=fixed_record_idx)
                if not renorm:
                    self.assertTrue(np.allclose(out[int_std],
                                    df.iloc[fixed_record_idx,
                                            df.columns.get_loc(int_std)])
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
                out = standardise_aggregate(df,
                                            renorm=renorm,
                                            fixed_record_idx=fixed_record_idx)
                if not renorm:
                    self.assertTrue(np.isclose(out.values,
                                    df.iloc[fixed_record_idx, :].values).any()
                                            )

class TestComplexStandardiseAggregate(unittest.TestCase):
    """Tests pandas complex internal standardisation aggregation method."""

    def setUp(self):
        self.mcols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.mdf = pd.DataFrame({k: v for k,v in zip(self.mcols,
                                np.random.rand(len(self.mcols), 10))})
        self.mdf = self.mdf.apply(lambda x: x/np.sum(x), axis='columns')

        self.tcols = ['SiO2', 'Ni', 'Cr', 'Sn']
        self.tdf = pd.DataFrame({k: v for k,v in zip(self.tcols,
                                np.random.rand(len(self.tcols), 10))})

        self.df = self.mdf.append(self.tdf, ignore_index=True, sort=False)

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = complex_standardise_aggregate(df, renorm=renorm)
                outvals = out.values[~np.isnan(out.values)]
                dfvals = df.values[~np.isnan(df.values)]
                self.assertTrue(np.allclose(outvals, dfvals))

    def test_multiple_with_IS(self):
        """
        Checks results on multiple records with internal standard specifed.
        """
        df = self.mdf.copy()
        fixed_record_idx = 0
        int_std = 'SiO2'
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = complex_standardise_aggregate(df,
                                            int_std=int_std,
                                            renorm=renorm,
                                            fixed_record_idx=fixed_record_idx)
                if not renorm:
                    self.assertTrue(np.allclose(out[int_std],
                                    df.iloc[fixed_record_idx,
                                            df.columns.get_loc(int_std)])
                                            )

    def test_multiple_without_IS(self):
        """
        Checks results on multiple records without internal standard specifed.
        """
        df = self.mdf
        out = complex_standardise_aggregate(df)

    def test_contrasting_with_IS(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for records which differ by all-but-one element
        df = self.df
        fixed_record_idx = 0
        int_std = 'SiO2'
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = complex_standardise_aggregate(df,
                                            int_std=int_std,
                                            renorm=renorm,
                                            fixed_record_idx=fixed_record_idx)
                if not renorm:
                    self.assertTrue(np.allclose(out[int_std],
                                    df.iloc[fixed_record_idx,
                                            df.columns.get_loc(int_std)])
                                            )

    def test_contrasting_without_IS(self):
        """
        Checks results on multiple contrasting records
        without internal standard specifed.
        """
        df = self.df
        out = complex_standardise_aggregate(df)


class TestNPComplexStandardiseAggregate(unittest.TestCase):
    """Tests numpy complex internal standardisation aggregation method."""

    def setUp(self):
        self.mcols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.mdf = pd.DataFrame({k: v for k,v in zip(self.mcols,
                                np.random.rand(len(self.mcols), 10))})
        self.mdf = self.mdf.apply(lambda x: x/np.sum(x), axis='columns')

        self.tcols = ['SiO2', 'Ni', 'Cr', 'Sn']
        self.tdf = pd.DataFrame({k: v for k,v in zip(self.tcols,
                                np.random.rand(len(self.tcols), 10))})

        self.df = self.mdf.append(self.tdf, ignore_index=True, sort=False)

    def test_single(self):
        """Checks results on single records."""
        df = self.df.head(1).copy()
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = np_complex_standardise_aggregate(df, renorm=renorm)
                outvals = out#[~np.isnan(out.values)]
                dfvals = df.values#[~np.isnan(df.values)]
                self.assertTrue(np.allclose(outvals, dfvals, equal_nan=True))

    def test_multiple_with_IS(self):
        """
        Checks results on multiple records with internal standard specifed.
        """
        df = self.mdf.copy()
        fixed_record_idx = 0
        int_std = 'SiO2'
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = np_complex_standardise_aggregate(df,
                                            int_std=int_std,
                                            renorm=renorm,
                                            fixed_record_idx=fixed_record_idx)
                if not renorm:
                    self.assertTrue(np.allclose(out[int_std],
                                    df.iloc[fixed_record_idx,
                                            df.columns.get_loc(int_std)])
                                            )

    def test_multiple_without_IS(self):
        """
        Checks results on multiple records without internal standard specifed.
        """
        df = self.mdf
        out = np_complex_standardise_aggregate(df)

    def test_contrasting_with_IS(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for records which differ by all-but-one element
        df = self.df
        fixed_record_idx = 0
        int_std = 'SiO2'
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                out = np_complex_standardise_aggregate(df,
                                            int_std=int_std,
                                            renorm=renorm,
                                            fixed_record_idx=fixed_record_idx)
                if not renorm:
                    self.assertTrue(np.allclose(out[int_std],
                                    df.iloc[fixed_record_idx,
                                            df.columns.get_loc(int_std)])
                                            )

    def test_contrasting_without_IS(self):
        """
        Checks results on multiple contrasting records
        without internal standard specifed.
        """
        df = self.df
        out = np_complex_standardise_aggregate(df)


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


class TestRenormalise(unittest.TestCase):
    """Tests the pandas renormalise utility."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.d = len(self.cols)
        self.n = 10
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(self.d, self.n))})

    def test_closure(self):
        """Checks whether closure is achieved."""
        df = self.df
        out = renormalise(df)
        self.assertTrue(np.allclose(out.sum(axis=1).values, 100.))

    def test_components_selection(self):
        """Checks partial closure for different sets of components."""
        pass


class TestALR(unittest.TestCase):
    """Test the numpy additive log ratio transformation."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})
        self.df = self.df.apply(lambda x: x/np.sum(x), axis='columns')

    def test_single(self):
        """Checks whether the function works on a single record."""
        df = self.df.head(1)
        out = alr(df.values)

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        df = self.df
        out = alr(df.values)

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        df = self.df.head(1)
        out = alr(df.values)
        inv = inv_alr(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        df = self.df
        out = alr(df.values)
        inv = inv_alr(out)
        self.assertTrue(np.allclose(inv, df.values))


class TestCLR(unittest.TestCase):
    """Test the centred log ratio transformation."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})
        self.df = self.df.apply(lambda x: x/np.sum(x), axis='columns')

    def test_single(self):
        """Checks whether the function works on a single record."""
        df = self.df.head(1)
        out = clr(df.values)

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        df = self.df
        out = clr(df.values)

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        df = self.df.head(1)
        out = clr(df.values)
        inv = inv_clr(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        df = self.df
        out = clr(df.values)
        inv = inv_clr(out)
        self.assertTrue(np.allclose(inv, df.values))


class TestOrthagonalBasis(unittest.TestCase):
    """Test the orthagonal basis generator for ILR transformation."""

    def test_orthagonal_basis(self):
        """Checks orthagonality of the transformation basis."""
        pass


class TestILR(unittest.TestCase):
    """Test the isometric log ratio transformation."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})
        self.df = self.df.apply(lambda x: x/np.sum(x), axis='columns')

    def test_single(self):
        """Checks whether the function works on a single record."""
        df = self.df.head(1)
        out = ilr(df.values)

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        df = self.df
        out = ilr(df.values)

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        df = self.df.head(1)
        out = ilr(df.values)
        inv = inv_ilr(out, X=df.values)
        self.assertTrue(np.allclose(inv, df.values))

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        df = self.df
        out = ilr(df.values)
        inv = inv_ilr(out, X=df.values)
        self.assertTrue(np.allclose(inv, df.values))


class TestLogTransformers(unittest.TestCase):
    """Checks the scikit-learn transformer classes."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})
        self.df = self.df.apply(lambda x: x/np.sum(x), axis='columns')

    def test_linear_transformer(self):
        """Test the linear transfomer."""
        df = self.df
        tmr = LinearTransform()
        out = tmr.transform(df.values)
        inv = tmr.inverse_transform(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_ALR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = ALRTransform()
        out = tmr.transform(df.values)
        inv = tmr.inverse_transform(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_CLR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = CLRTransform()
        out = tmr.transform(df.values)
        inv = tmr.inverse_transform(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_ILR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = ILRTransform()
        out = tmr.transform(df.values)
        inv = tmr.inverse_transform(out)
        self.assertTrue(np.allclose(inv, df.values))


if __name__ == '__main__':
    unittest.main()
