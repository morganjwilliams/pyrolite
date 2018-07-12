import unittest
import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
import ternary
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as matax

from pyrolite.geochem import REE, common_elements
from pyrolite.plot import ternaryplot, spiderplot, densityplot

import logging
logging.getLogger(__name__)

class TestSpiderplot(unittest.TestCase):
    """Tests the Spiderplot functionality."""

    def setUp(self):
        reels = REE(output='string')
        self.df = pd.DataFrame({k: v for k,v in zip(reels,
                                np.random.rand(len(reels), 10))})

    def test_none(self):
        """Test generation of plot with no data."""
        pass

    def test_one(self):
        """Test generation of plot with one record."""
        pass

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        pass

    def test_no_axis_specified(self):
        """Test generation of plot without axis specified."""
        pass

    def test_axis_specified(self):
        """Test generation of plot with axis specified."""
        pass

    def test_no_components_specified(self):
        """Test generation of plot with no components specified."""
        pass

    def test_components_specified(self):
        """Test generation of plot with components specified."""
        pass

    def test_plot_off(self):
        """Test plot generation with plot off."""
        pass

    def test_fill(self):
        """Test fill functionality is available."""
        pass

    @unittest.expectedFailure
    def test_noplot_nofill(self):
        """Test failure on no-plot no-fill options."""
        out = spiderplot(self.df, plot=False, fill=False)
        self.assertTrue(isinstance(out, Maxes.Axes))
        plt.close('all')

    def test_valid_style(self):
        """Test valid styling options."""
        pass

    def test_log_on_irrellevant_style_options(self):
        """Test stability under additional kwargs."""
        style = {'thingwhichisnotacolor': 'notacolor', 'irrelevant': 'red'}
        with self.assertLogs(level='INFO') as cm:
            #with self.assertWarns(UserWarning):
            ax = spiderplot(self.df, **style)

        plt.close('all')

    @unittest.expectedFailure
    def test_invalid_style_options(self):
        """Test stability under invalid style values."""
        style = {'color': 'notacolor', 'marker': 'red'}
        spiderplot(self.df, **style)
        plt.close('all')

    def tearDown(self):
        plt.close('all')

class TestTernaryplot(unittest.TestCase):
    """Tests the Ternaryplot functionality."""

    def setUp(self):
        self.cols = ['MgO', 'CaO', 'SiO2']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_none(self):
        """Test generation of plot with no data."""
        df = pd.DataFrame(columns=self.cols)
        out = ternaryplot(df)
        self.assertEqual(type(out),
                         ternary.ternary_axes_subplot.TernaryAxesSubplot)
        plt.close('all')

    def test_one(self):
        """Test generation of plot with one record."""
        df = self.df.head(1)
        out = ternaryplot(df)
        self.assertEqual(type(out),
                         ternary.ternary_axes_subplot.TernaryAxesSubplot)
        plt.close('all')

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        df = self.df.loc[:, :]
        out = ternaryplot(df)
        self.assertEqual(type(out),
                         ternary.ternary_axes_subplot.TernaryAxesSubplot)
        plt.close('all')

    def test_overplotting(self):
        """Test use of the plot for multiple rounds of plotting."""
        pass

    def tearDown(self):
        plt.close('all')


class TestDensityplot(unittest.TestCase):
    """Tests the Densityplot functionality."""

    def setUp(self):
        self.cols = ['MgO', 'SiO2', 'CaO']
        data = np.array([0.5, 0.4, 0.3])
        cov =   np.array([[2, -1, -0.5],
                         [-1, 2, -1],
                         [-0.5, -1, 2]])
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
                out = densityplot(df)
                self.assertTrue(isinstance(out, matax.Axes))
                plt.close('all')


    def test_one(self):
        """Test generation of plot with one record."""

        for df in [self.bidf.head(1), self.tridf.head(1)]:
            with self.subTest(df=df):
                out = densityplot(self.bidf)
                self.assertTrue(isinstance(out, matax.Axes))
                plt.close('all')

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        for df in [self.bidf, self.tridf]:
            with self.subTest(df=df):
                out = densityplot(df)
                self.assertTrue(isinstance(out, matax.Axes))
                plt.close('all')


    def test_modes(self): #
        """Tests different ploting modes."""
        for df in [self.bidf, self.tridf]:
            with self.subTest(df=df):
                for mode in ['density', 'hist2d', 'hexbin']:
                    with self.subTest(mode=mode):
                        out = densityplot(df, mode=mode)
                        self.assertTrue(isinstance(out, matax.Axes))
                        plt.close('all')

    def test_bivariate_logscale(self): #
        """Tests logscale for different ploting modes using bivariate data."""
        df = self.bidf
        for logspace in [True, False]:
            with self.subTest(logspace=logspace):
                for mode in ['density', 'hist2d', 'hexbin']:
                    with self.subTest(mode=mode):
                        out = densityplot(df, mode=mode)
                        self.assertTrue(isinstance(out, matax.Axes))
                        plt.close('all')


    def test_overplotting(self):
        """Test use of the plot for multiple rounds of plotting."""
        pass

    def tearDown(self):
        plt.close('all')

if __name__ == '__main__':
    unittest.main()
