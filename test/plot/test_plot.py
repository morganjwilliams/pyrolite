import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
import logging
from pyrolite.plot import tern, spider, density, pyroplot
from pyrolite.geochem import REE
from pyrolite.util.synthetic import random_composition

logger = logging.getLogger(__name__)


class TestPyroPlot(unittest.TestCase):
    def setUp(self):
        self.cols = ["MgO", "SiO2", "CaO"]
        data = np.array([0.5, 0.4, 0.3])

        self.bidf = pd.DataFrame(
            data=random_composition(mean=data[:2], size=100),
            columns=self.cols[:2],
        )

        self.tridf = pd.DataFrame(
            data=random_composition(mean=data,size=100), columns=self.cols
        )

        ree = REE()
        self.multidf = pd.DataFrame(
            data=random_composition(size=100, D=len(ree)), columns=ree
        )

        # add a small number of nans
        self.bidf.iloc[0, 1] = np.nan
        self.tridf.iloc[0, 1] = np.nan
        self.multidf.iloc[0, 1] = np.nan

    def test_pyroplot_class_structure(self):
        pplot = pyroplot  # class
        for attr in ["spider", "ternary", "density", "REE"]:
            self.assertTrue(hasattr(pplot, attr))

    def test_pyroplot_instance_structure(self):
        pplot = pyroplot(self.bidf)  # instance
        for attr in ["spider", "ternary", "density", "REE"]:
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

    @unittest.expectedFailure
    def test_ternary_with_two_components(self):
        self.bidf.pyroplot.ternary()

    @unittest.expectedFailure
    def test_ternary_with_more_components(self):
        self.multidf.pyroplot.ternary()

    def test_ternary_with_more_components_specified(self):
        self.multidf.pyroplot.ternary(components=self.multidf.columns[:3])

    def test_density(self):
        self.bidf.pyroplot.density()

    def test_density_ternary(self):
        self.tridf.pyroplot.density()

    @unittest.expectedFailure
    def test_density_with_more_components(self):
        self.multidf.pyroplot.density()

    def test_density_with_more_components_specified(self):
        self.multidf.pyroplot.density(components=self.multidf.columns[:2])

    def test_density_with_more_components_specified_ternary(self):
        self.multidf.pyroplot.density(components=self.multidf.columns[:3])

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
