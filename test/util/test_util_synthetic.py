import unittest
import pandas as pd
import numpy as np
from pyrolite.util.synthetic import *


class TestRandomCovMatrix(unittest.TestCase):
    """
    Check that the random covariance matrix produces a symmetric postive-semidefinite
    covariance matrix.
    """

    def test_shape(self):
        for shape in [2, 5]:
            mat = random_cov_matrix(shape)
            self.assertTrue(mat.shape == (shape, shape))  # shape
            self.assertTrue(np.allclose(mat, mat.T))  # symmetry
            for i in range(shape):
                self.assertTrue(
                    np.linalg.det(mat[0:i, 0:i]) > 0.0
                )  # sylvesters criterion

    def test_validate(self):
        for validate in [True, False]:
            mat = random_cov_matrix(3, validate=validate)


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

    def test_missing_mechanism(self):
        for missing in [None, "MCAR", "MAR", "MNAR"]:
            random_composition(size=self.size, D=self.D, missing=missing)

    def test_missing_mechanism_invalid(self):
        for missing in ["all", "completely"]:
            with self.assertRaises(NotImplementedError):
                random_composition(size=self.size, D=self.D, missing=missing)

    def test_missing_propnan(self):
        pass


class TestRandomCovMatrix(unittest.TestCase):
    """
    Check that the random covariance matrix produces a symmetric postive-semidefinite
    covariance matrix.
    """

    def test_shape(self):
        for shape in [2, 5]:
            mat = random_cov_matrix(shape)
            self.assertTrue(mat.shape == (shape, shape))  # shape
            self.assertTrue(np.allclose(mat, mat.T))  # symmetry
            for i in range(shape):
                self.assertTrue(
                    np.linalg.det(mat[0:i, 0:i]) > 0.0
                )  # sylvesters criterion


if __name__ == "__main__":
    unittest.main()
