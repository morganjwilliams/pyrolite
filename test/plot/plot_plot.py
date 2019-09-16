import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
import logging
from pyrolite.plot import tern, spider, density, pyroplot
from pyrolite.geochem import REE
from pyrolite.util.synthetic import test_df

logger = logging.getLogger(__name__)


class TestPyroPlot(unittest.TestCase):
    def setUp(self):
        self.cols = ["MgO", "SiO2", "CaO"]

        # can run into interesting singular matrix errors with bivariate random data
        self.tridf = test_df(cols=self.cols, index_length=100)
        self.bidf = self.tridf.loc[:, self.cols[:2]]
        self.multidf = test_df(cols=REE(), index_length=100)

        # add a small number of nans
        self.bidf.iloc[0, 1] = np.nan
        self.tridf.iloc[0, 1] = np.nan
        self.multidf.iloc[0, 1] = np.nan

    def tearDown(self):
        plt.close("all")

    def test_pyroplot_class_structure(self):
        pplot = pyroplot  # class
        for attr in ["spider", "ternary", "density", "REE", "stem", "parallel"]:
            self.assertTrue(hasattr(pplot, attr))

    def test_pyroplot_instance_structure(self):
        pplot = pyroplot(self.bidf)  # instance
        for attr in ["spider", "ternary", "density", "REE", "stem", "parallel"]:
            self.assertTrue(hasattr(pplot, attr))

    def test_pandas_api_accessor_exists(self):
        for pdobj in [self.bidf, self.tridf, self.multidf]:
            self.assertTrue(hasattr(pdobj, "pyroplot"))

    def test_cooccurence_default(self):
        self.multidf.pyroplot.cooccurence()

    def test_cooccurence_normalize(self):
        for normalize in [True, False]:
            with self.subTest(normalize=normalize):
                self.multidf.pyroplot.cooccurence(normalize=normalize)

    def test_cooccurence_log(self):
        for log in [True, False]:
            with self.subTest(log=log):
                self.multidf.pyroplot.cooccurence(log=log)

    def test_cooccurence_colorbar(self):
        for colorbar in [True, False]:
            with self.subTest(colorbar=colorbar):
                self.multidf.pyroplot.cooccurence(colorbar=colorbar)

    def test_cooccurencet_external_ax(self):
        fig, ax = plt.subplots(1)
        self.multidf.pyroplot.cooccurence(ax=ax)

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

    def test_ree(self):
        self.multidf.pyroplot.REE()

    def test_parallel(self):
        self.tridf.pyroplot.parallel()

    def test_scatter_default(self):
        self.bidf.pyroplot.scatter()

    def test_spider(self):
        self.multidf.pyroplot.spider()

    def test_stem_default(self):
        self.bidf.pyroplot.stem()

    def test_stem_v(self):
        self.bidf.pyroplot.stem(orientation="V")

    def test_ternary(self):
        self.tridf.pyroplot.ternary()

    def test_ternary_labels(self):
        for labels in [[], self.tridf.columns.tolist()]:
            with self.subTest(labels=labels):
                self.tridf.pyroplot.ternary(axlabels=labels)

    @unittest.expectedFailure
    def test_ternary_with_two_components(self):
        self.bidf.pyroplot.ternary()

    @unittest.expectedFailure
    def test_ternary_with_more_components(self):
        self.multidf.pyroplot.ternary()

    def test_ternary_with_more_components_specified(self):
        self.multidf.pyroplot.ternary(components=self.multidf.columns[:3])


if __name__ == "__main__":
    unittest.main()
