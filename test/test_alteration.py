import unittest
import numpy as np
from pyrolite.alteration import *


class TestCIA(unittest.TestCase):
    """Tests the chemical index of alteration measure."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO',
                     'TiO2', 'Na2O', 'K2O', 'Al2O3']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_CIA(self):
        df = self.df
        df.loc[:, 'CIA'] = CIA(df)

class TestCIW(unittest.TestCase):
    """Tests the chemical index of weathering measure."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO',
                     'TiO2', 'Na2O', 'K2O', 'Al2O3']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_CIW(self):
        df = self.df
        df.loc[:, 'CIW'] = CIW(df)

class TestPIA(unittest.TestCase):
    """Tests the plagioclase index of alteration measure."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO',
                     'TiO2', 'Na2O', 'K2O', 'Al2O3']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_PIA(self):
        df = self.df
        df.loc[:, 'PIA'] = PIA(df)


class TestSAR(unittest.TestCase):
    """Tests the silica alumina ratio measure."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO',
                     'TiO2', 'Na2O', 'K2O', 'Al2O3']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_SAR(self):
        df = self.df
        df.loc[:, 'SAR'] = SAR(df)


class TestSiTiIndex(unittest.TestCase):
    """Tests the silica titania ratio measure."""

    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO',
                     'TiO2', 'Na2O', 'K2O', 'Al2O3']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_SiTiIndex(self):
        df = self.df
        df.loc[:, 'SiTiIndex'] = SiTiIndex(df)


class TestWIP(unittest.TestCase):
    """Tests the weathering index of parker measure."""
    def setUp(self):
        self.cols = ['SiO2', 'CaO', 'MgO', 'FeO',
                     'TiO2', 'Na2O', 'K2O', 'Al2O3']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_WIP(self):
        df = self.df
        df.loc[:, 'WIP'] = WIP(df)


if __name__ == '__main__':
    unittest.main()
