import unittest
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
import logging

logger = logging.getLogger(__name__)


class TestDensityplot(unittest.TestCase):
    """Tests the Densityplot functionality."""

    def setUp(self):
        self.cols = ["MgO", "SiO2", "CaO"]
        data = np.array([0.5, 0.4, 0.3])
        cov = np.array([[2, -1, -0.5], [-1, 2, -1], [-0.5, -1, 2]])
        bidata = multivariate_normal(data[:2], cov[:2, :2], 2000)
        bidata[0, 1] = np.nan
        self.bidf = pd.DataFrame(bidata, columns=self.cols[:2])
        tridata = multivariate_normal(data, cov, 2000)
        bidata[0, 1] = np.nan
        self.tridf = pd.DataFrame(tridata, columns=self.cols)

    def test_none(self):
        """Test generation of plot with no data."""
        for df in [pd.DataFrame(columns=self.cols)]:
            with self.subTest(df=df):
                out = density.density(df)
                self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                plt.close("all")

    def test_one(self):
        """Test generation of plot with one record."""

        for df in [self.bidf.head(1), self.tridf.head(1)]:
            with self.subTest(df=df):
                out = density.density(self.bidf)
                self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                plt.close("all")

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        for df in [self.bidf, self.tridf]:
            with self.subTest(df=df):
                out = density.density(df)
                self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                plt.close("all")

    def test_modes(self):  #
        """Tests different ploting modes."""
        for df in [self.bidf, self.tridf]:
            with self.subTest(df=df):
                for mode in ["density", "hist2d", "hexbin"]:
                    with self.subTest(mode=mode):
                        out = density.density(df, mode=mode)
                        self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                        plt.close("all")

    def test_bivariate_logscale(self):  #
        """Tests logscale for different ploting modes using bivariate data."""
        df = self.bidf
        df[df<0] = np.nan
        for logspacing in [(True, True), (False, False), (False, True), (True, False)]:
            lx, ly = logspacing
            with self.subTest(logx=lx, logy=ly):
                for mode in ["density", "hist2d", "hexbin"]:
                    with self.subTest(mode=mode):
                        out = density.density(df, mode=mode, logx=lx, logy=ly)
                        self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                        plt.close("all")

    def tearDown(self):
        plt.close("all")



if __name__ == "__main__":
    unittest.main()
