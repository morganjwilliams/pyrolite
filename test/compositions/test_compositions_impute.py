import unittest
import numpy as np
import pandas as pd
from pyrolite.util.pd import test_df, test_ser
from pyrolite.compositions.aggregate import np_cross_ratios
from pyrolite.compositions.impute import *


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


if __name__ == '__main__':
    unittest.main()
