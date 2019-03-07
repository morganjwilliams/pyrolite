import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
import logging
from pyrolite.plot import ternary, spider, density, pyroplot
from pyrolite.geochem import REE
from pyrolite.util.math import random_cov_matrix

logger = logging.getLogger(__name__)


class TestPyroPlot(unittest.TestCase):
    def setUp(self):
        self.cols = ["MgO", "SiO2", "CaO"]
        data = np.array([0.5, 0.4, 0.3])
        cov = np.array([[2, -1, -0.5], [-1, 2, -1], [-0.5, -1, 2]])

        self.bidf = pd.DataFrame(
            multivariate_normal(data[:2], cov[:2, :2], 100), columns=self.cols[:2]
        )

        self.tridf = pd.DataFrame(
            multivariate_normal(data, cov, 100), columns=self.cols
        )

        ree = REE()
        cov_ree = random_cov_matrix(len(ree))
        self.multidf = pd.DataFrame(
            multivariate_normal(np.random.randn(len(ree)), cov_ree, 100), columns=ree
        )

        # add a small number of nans
        self.bidf.iloc[0, 1] = np.nan
        self.tridf.iloc[0, 1] = np.nan
        self.multidf.iloc[0, 1] = np.nan

    def test_pyroplot_class_structure(self):
        pplot = pyroplot # class
        for attr in ['spider', 'ternary', 'density', 'REE']:
            self.assertTrue(hasattr(pplot, attr))

    def test_pyroplot_instance_structure(self):
        pplot = pyroplot(self.bidf) # instance
        for attr in ['spider', 'ternary', 'density', 'REE']:
            self.assertTrue(hasattr(pplot, attr))

    def test_pandas_api_accessor_exists(self):
        for pdobj in [self.bidf, self.tridf, self.multidf]:
            self.assertTrue(hasattr(pdobj, "pyroplot"))

    def test_spider(self):
        self.multidf.pyroplot.spider()

    def test_ree(self):
        self.multidf.pyroplot.REE()

    def test_ternary(self):
        self.tridf.pyroplot.ternary()

    def test_density(self):
        self.bidf.pyroplot.density()

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
