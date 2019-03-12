import unittest
import numpy as np
import pandas as pd
from pyrolite.util.pd import test_df, test_ser
from pyrolite.comp.aggregate import np_cross_ratios
from pyrolite.util.math import random_cov_matrix
from pyrolite.comp.impute import *


class TestAugmentedCovarianceMatrix(unittest.TestCase):
    def setUp(self):
        self.mean = np.random.randn(5)
        self.cov = random_cov_matrix(5)

    def test_augmented_covariance_matrix(self):
        ACM = augmented_covariance_matrix(self.mean, self.cov)


class TestMDPattern(unittest.TestCase):
    def setUp(self):
        self.data = random_composition_missing(size=200, MAR=False)

    def test_md_pattern(self):
        pattern_ids, PD = md_pattern(self.data)


class TestEMCOMP(unittest.TestCase):
    def setUp(self):
        self.data = random_composition_missing(size=200, MAR=False)

    def test_encomp(self):
        impute, p0, ni = EMCOMP(
            self.data, threshold=0.1 * np.nanmin(self.data, axis=0), tol=0.01
        )


class TestPDImputeRatios(unittest.TestCase):
    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
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
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.d = len(self.cols)
        self.n = 10
        self.df = test_df(cols=self.cols, index_length=self.n)

    def test_imputation(self):
        """Checks results on single record."""
        df = self.df.head(1).copy()
        n = df.index.size
        df.iloc[:, np.random.randint(1, self.d, size=1)] = np.nan
        arr = df.values  # single array
        ratios = np_cross_ratios(arr)[0]
        imputed = np_impute_ratios(ratios)


if __name__ == "__main__":
    unittest.main()
