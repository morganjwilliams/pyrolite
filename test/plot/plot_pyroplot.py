import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
from pyrolite.plot import spider, density, pyroplot
from pyrolite.geochem import REE
from pyrolite.util.synthetic import normal_frame
import matplotlib.colors
import mpltern.ternary

import logging

logger = logging.getLogger(__name__)

np.random.seed(81)


class TestPyroPlot(unittest.TestCase):
    def setUp(self):
        self.cols = ["MgO", "SiO2", "CaO"]

        # can run into interesting singular matrix errors with bivariate random data
        self.tridf = normal_frame(columns=self.cols, size=100)
        self.bidf = self.tridf.loc[:, self.cols[:2]]
        self.multidf = normal_frame(columns=REE(), size=100)

        # add a small number of nans
        self.bidf.iloc[0, 1] = np.nan
        self.tridf.iloc[0, 1] = np.nan
        self.multidf.iloc[0, 1] = np.nan

    def tearDown(self):
        plt.close("all")

    def test_pyroplot_class_structure(self):
        pplot = pyroplot  # class
        for attr in ["spider", "scatter", "plot", "density", "REE", "stem", "parallel"]:
            self.assertTrue(hasattr(pplot, attr))

    def test_pyroplot_instance_structure(self):
        pplot = pyroplot(self.bidf)  # instance
        for attr in ["spider", "scatter", "plot", "density", "REE", "stem", "parallel"]:
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

    def test_heatscatter(self):
        self.bidf.pyroplot.heatscatter()

    def test_heatscatter_ternary(self):
        self.tridf.pyroplot.heatscatter()

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

    def test_scatter_ternary(self):
        self.tridf.pyroplot.scatter()

    def test_scatter_ternary_labels(self):
        for labels in [False, True]:
            with self.subTest(labels=labels):
                self.tridf.pyroplot.scatter(axlabels=labels)

    @unittest.expectedFailure
    def test_scatter_with_more_components(self):
        self.multidf.pyroplot.scatter()

    def test_scatter_with_more_components_specified(self):
        self.multidf.pyroplot.scatter(components=self.multidf.columns[:3])


class TestPyroTernary(unittest.TestCase):
    def setUp(self):
        self.cols = ["MgO", "SiO2", "CaO"]

        # can run into interesting singular matrix errors with bivariate random data
        self.tridf = normal_frame(columns=self.cols, size=100)

    def test_default(self):
        """Test generation of plot with one record."""
        ax = self.tridf.pyroplot.scatter()
        self.assertIsInstance(ax, mpltern.ternary.TernaryAxes)

    def test_overplotting(self):
        """Test use of the plot for multiple rounds of plotting."""
        ax1 = self.tridf.pyroplot.scatter()
        ax2 = self.tridf.pyroplot.scatter(ax=ax1)
        self.assertIsInstance(ax2, mpltern.ternary.TernaryAxes)
        self.assertTrue(ax1 is ax2)  # hasn't added a new ternary axis

    def test_color_tuple(self):
        ax = self.tridf.pyroplot.scatter(c=(0.1, 0.2, 0.5, 0.3))

    def test_color_hex(self):
        ax = self.tridf.pyroplot.scatter(c="#0f0f0f")

    def test_color_cmap_c_combination(self):
        """
        Check than array of values specified using `c`
        can be used for a colormap.
        """
        ax = self.tridf.pyroplot.scatter(c=np.linspace(0, 10, 100), cmap="viridis")

    def test_norm_specified(self):
        ax = self.tridf.pyroplot.scatter(
            c=np.random.randn(100),
            cmap="viridis",
            norm=matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0),
        )

    def test_label_specified(self):
        ax = self.tridf.pyroplot.scatter(label="testarr")

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
