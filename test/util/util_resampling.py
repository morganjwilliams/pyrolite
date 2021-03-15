import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.util.resampling import (
    _segmented_univariate_distance_matrix,
    univariate_distance_matrix,
    get_spatiotemporal_resampling_weights,
    add_age_noise,
    spatiotemporal_bootstrap_resample,
)

from pyrolite.util.spatial import great_circle_distance
from pyrolite.util.synthetic import normal_frame

df = normal_frame()

sample = df.sample(frac=1, replace=True)
dir(df.index)
df.index.take(sample.index)
df.index.intersection(sample.index)


def _get_spatiotemporal_dataframe(size, geochem_columns=[]):
    df = pd.DataFrame(
        index=range(size),
        columns=["Latitude", "Longitude", "Age", "AgeUncertainty", "MinAge", "MaxAge",],
    )
    df["Latitude"] = 20 + np.random.randn(size)
    df["Longitude"] = 85 + np.random.randn(size)
    df["Age"] = 100 + np.random.randint(0, 100, size)
    df["AgeUncertainty"] = 2 + np.random.randn(size)
    df["MinAge"] = df["Age"] - 2
    df["MaxAge"] = df["Age"] + 2

    if geochem_columns:
        df[geochem_columns] = normal_frame(columns=geochem_columns, size=size)
    return df


class TestSpatioTemporalBoostrapResample(unittest.TestCase):
    # TODO: variations of uncertainty (0D, 1D, 2D uncertainties)
    # TODO: add transform for non-resampled parameters
    def setUp(self):
        self.geochem_columns = ["SiO2", "Ti", "Al2O3"]
        self.df = _get_spatiotemporal_dataframe(
            10, geochem_columns=self.geochem_columns
        )

    def test_default(self):
        _df = self.df
        output = spatiotemporal_bootstrap_resample(
            _df, columns=self.geochem_columns, niter=10
        )
        self.assertIsInstance(output, dict)

    def test_categories(self):
        _df = self.df
        _df["Grouping"] = np.random.randint(3, size=_df.index.size)
        output = spatiotemporal_bootstrap_resample(
            _df, columns=self.geochem_columns, niter=10, categories="Grouping"
        )
        self.assertIsInstance(output, dict)

    def test_uncertainty_modes(self):
        _df = self.df
        subset_shape = _df.index.size, len(self.geochem_columns)
        for uncert in [
            0.5,
            np.random.uniform(size=subset_shape[1]),
            np.random.uniform(size=subset_shape),
        ]:
            with self.subTest(uncert=uncert):
                output = spatiotemporal_bootstrap_resample(
                    _df, columns=self.geochem_columns, uncert=uncert, niter=5
                )

    def test_transform(self):
        _df = self.df
        output = spatiotemporal_bootstrap_resample(_df, niter=10, transform=np.log)


class TestGetSpatiotemporalResamplingWeights(unittest.TestCase):
    def setUp(self):
        self.df = _get_spatiotemporal_dataframe(10)

    def test_default(self):
        _df = self.df
        weights = get_spatiotemporal_resampling_weights(_df)
        self.assertTrue(weights.ndim == 1)
        self.assertTrue(np.isclose(weights.sum(), 1.0))
        self.assertTrue(weights.size == _df.index.size)


class TestUnivariateDistanceMatrix(unittest.TestCase):
    def setUp(self):
        self.df = _get_spatiotemporal_dataframe(10)

    def test_default(self):
        """
        Here we check that the function returns an array of the appropriate shape, size
        and which has expected properties for distances (>=0).
        """
        _df = self.df
        age_distances = univariate_distance_matrix(_df["Age"])
        self.assertTrue(age_distances.ndim == 2)
        self.assertTrue(age_distances.shape[0] == age_distances.shape[1])
        self.assertTrue(age_distances.shape[0] == _df.index.size)
        self.assertTrue((age_distances >= 0).all())
        self.assertTrue((np.diag(age_distances) == 0).all())


class TestAddAgeNoise(unittest.TestCase):
    def setUp(self):
        self.df = _get_spatiotemporal_dataframe(10)

    def test_default(self):
        # copy the dataframe to see what it's like before editing
        _df = self.df.copy(deep=True)
        output = add_age_noise(_df)
        self.assertTrue((_df.columns == output.columns).all())
        self.assertTrue((self.df["Age"] != output["Age"]).all())

    def test_minmax_only(self):
        _df = self.df.copy(deep=True)
        _df.drop(columns=["AgeUncertainty"], inplace=True)
        output = add_age_noise(_df)
        self.assertTrue((_df.columns == output.columns).all())
        self.assertTrue((self.df["Age"] != output["Age"]).all())

    @unittest.expectedFailure
    def test_no_uncertainties_available(self):
        _df = self.df.copy(deep=True)
        _df.drop(columns=["AgeUncertainty", "MaxAge"], inplace=True)
        output = add_age_noise(_df)
        self.assertTrue((_df.columns == output.columns).all())
        self.assertTrue((self.df["Age"] != output["Age"]).all())

    def test_nearzero_ages(self):
        """
        Ages shouldn't go into the future...?
        """
        pass


if __name__ == "__main__":
    unittest.main()
