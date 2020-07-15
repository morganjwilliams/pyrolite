import unittest
import pandas as pd
import numpy as np
from pyrolite.util.synthetic import *


class TestExampleSpiderData(unittest.TestCase):
    def test_default(self):
        size = 20
        df = example_spider_data(size=size)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Eu", df.columns)

    def test_norm_None(self):
        size = 20
        df = example_spider_data(size=size, norm_to=None)
        df2 = example_spider_data(size=size)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Eu", df.columns)
        self.assertTrue(((df2["Cs"] / df["Cs"]) > 1).all())


class TestRandomCovMatrix(unittest.TestCase):
    """
    Check that the random covariance matrix produces a symmetric postive-semidefinite
    covariance matrix.
    """

    def test_shape(self):
        """Check that a covariance matrix of arbitrary size can be generated."""
        for shape in [1, 2, 5]:
            with self.subTest(shape=shape):
                mat = random_cov_matrix(shape)
                self.assertTrue(mat.shape == (shape, shape))  # shape
                self.assertTrue(np.allclose(mat, mat.T))  # symmetry
                for i in range(shape):
                    self.assertTrue(
                        np.linalg.det(mat[0:i, 0:i]) > 0.0
                    )  # sylvesters criterion

    def test_validate(self):
        """Check that the covariance matrix can be checked at creation time."""
        shape = 3
        for validate in [True, False]:
            with self.subTest(validate=validate):
                mat = random_cov_matrix(shape, validate=validate)
                self.assertTrue(mat.shape == (shape, shape))  # shape
                self.assertTrue(np.allclose(mat, mat.T))  # symmetry
                for i in range(shape):
                    self.assertTrue(
                        np.linalg.det(mat[0:i, 0:i]) > 0.0
                    )  # sylvesters criterion


class TestRandomComposition(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.D = 4

    def test_default(self):
        rc = random_composition(size=self.size, D=self.D)

    def test_size(self):
        for size in [1, 10, 100]:
            rc = random_composition(size=size, D=self.D)

    def test_D(self):
        for D in [1, 3, 5]:
            rc = random_composition(size=self.size, D=D)

    def test_mean_specified(self):
        mean = random_composition(size=1, D=self.D)
        rc = random_composition(size=self.size, D=self.D, mean=mean)

    def test_cov_specified(self):
        cov = random_cov_matrix(self.D - 1)
        rc = random_composition(size=self.size, D=self.D, cov=cov)

    def test_mean_cov_specified(self):
        mean = random_composition(size=1, D=self.D)
        cov = random_cov_matrix(self.D - 1)
        rc = random_composition(size=self.size, D=self.D, mean=mean, cov=cov)

    def test_missing_mechanism(self):
        for missing in [None, "MCAR", "MAR", "MNAR"]:
            rc = random_composition(size=self.size, D=self.D, missing=missing)

    def test_missing_columns(self):
        for missing_columns in [1, 2, (0, 1), [1, 2]]:
            with self.subTest(missing_columns=missing_columns):
                rc = random_composition(
                    size=self.size, D=self.D, missing_columns=missing_columns
                )

    def test_missing_mechanism_invalid(self):
        for missing in ["all", "completely"]:
            with self.assertRaises(NotImplementedError):
                random_composition(size=self.size, D=self.D, missing=missing)

    def test_missing_propnan(self):
        for propnan in [0, 0.05, 0.3]:
            with self.subTest(propnan=propnan):
                rc = random_composition(
                    size=self.size, D=self.D, propnan=propnan, missing="MCAR"
                )


if __name__ == "__main__":
    unittest.main()
