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


if __name__ == "__main__":
    unittest.main()
