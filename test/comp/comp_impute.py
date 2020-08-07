import unittest
import numpy as np
import pandas as pd
from pyrolite.util.math import augmented_covariance_matrix
from pyrolite.util.synthetic import (
    normal_frame,
    normal_series,
    random_cov_matrix,
    random_composition,
)
from pyrolite.comp.aggregate import np_cross_ratios
from pyrolite.comp.impute import (
    EMCOMP,
    _little_sweep,
    _reg_sweep,
    _multisweep
)


class TestRegSweep(unittest.TestCase):
    def setUp(self):
        self.G = augmented_covariance_matrix(np.array([1.1, 0.9]), random_cov_matrix(2))
        self.G3 = augmented_covariance_matrix(
            np.array([1.1, 0.9, 1.05]), random_cov_matrix(3)
        )

    def test_default(self):
        pass


class TestLittleSweep(unittest.TestCase):
    def setUp(self):
        self.G = augmented_covariance_matrix(np.array([1.1, 0.9]), random_cov_matrix(2))
        self.G3 = augmented_covariance_matrix(
            np.array([1.1, 0.9, 1.05]), random_cov_matrix(3)
        )

    def test_multisweep_commutative(self):
        G = self.G3
        assert np.allclose(_multisweep(G, [0, 1]), _multisweep(G, [1, 0]))

    def test_default(self):
        G = self.G
        H = _little_sweep(G, k=0)

        assert np.allclose(
            H,
            np.array(
                [
                    [-1 / G[0, 0], G[0, 1] / G[0, 0], G[0, 2] / G[0, 0]],
                    [
                        G[0, 1] / G[0, 0],
                        G[1, 1] - G[0, 1] * G[0, 1] / G[0, 0],
                        G[1, 2] - G[0, 2] * G[0, 1] / G[0, 0],
                    ],
                    [
                        G[0, 2] / G[0, 0],
                        G[1, 2] - G[0, 2] * G[0, 1] / G[0, 0],
                        G[2, 2] - G[0, 2] * G[0, 2] / G[0, 0],
                    ],
                ]
            ),
        )


class TestEMCOMP(unittest.TestCase):
    def setUp(self):
        self.data = random_composition(size=200, missing="MNAR")

    def test_encomp(self):
        impute, p0, ni = EMCOMP(
            self.data, threshold=0.5 * np.nanmin(self.data, axis=0), tol=0.01
        )


if __name__ == "__main__":
    unittest.main()
