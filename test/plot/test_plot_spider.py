import unittest
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
from pyrolite.geochem import REE
from pyrolite.plot.spider import spider, REE_v_radii
import logging

logger = logging.getLogger(__name__)


class TestSpiderplot(unittest.TestCase):
    """Tests the Spiderplot functionality."""

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.els = REE()
        self.arr = np.random.rand(10, len(self.els))


    def test_none(self):
        """Test generation of plot with no data."""
        ax = spider(np.empty(0))
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

    def test_one(self):
        """Test generation of plot with one record."""
        pass

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        pass

    def test_axis_specified(self):
        """Test generation of plot with axis specified."""
        ax = spider(self.arr, ax=self.ax)
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

    def test_fill(self):
        """Test fill functionality is available."""
        for plot, fill in [(False, True), (True, True)]:
            ax = spider(self.arr, plot=plot, fill=fill)
            self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

    @unittest.expectedFailure
    def test_noplot_nofill(self):
        """Test failure on no-plot no-fill options."""
        for arr in [self.arr]:
            ax = spider(arr, plot=False, fill=False)

    def test_valid_style(self):
        """Test valid styling options."""
        pass

    def test_log_on_irrellevant_style_options(self):
        """Test stability under additional kwargs."""
        style = {"thingwhichisnotacolor": "notacolor", "irrelevant": "red"}
        with self.assertLogs(level="INFO") as cm:
            # with self.assertWarns(UserWarning):
            for arr in [self.arr]:
                ax = spider(arr, **style)


    @unittest.expectedFailure
    def test_invalid_style_options(self):
        """Test stability under invalid style values."""
        style = {"color": "notacolor", "marker": "red"}
        for arr in [self.arr]:
            ax = spider(arr, **style)

    def tearDown(self):
        plt.close("all")


class TestREERadiiPlot(unittest.TestCase):
    """Tests the REE_radii_plot functionality."""

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.reels = REE()
        self.arr = np.random.rand(10, len(self.reels))

    def test_none(self):
        """Test generation of plot with no data."""
        for arr in [np.empty(0), None]:
            with self.subTest(arr=arr):
                ax = REE_v_radii(arr=arr)

    def test_one(self):
        ax = REE_v_radii(self.arr[0, :], ree=self.reels)

    def test_default(self):
        for arr in [self.arr]:
            ax = REE_v_radii(arr, ree=self.reels)

    def test_mode(self):
        for mode in ['radii', 'elements']:
            ax = REE_v_radii(self.arr, ree=self.reels)

    def test_external_ax(self):
        ax = REE_v_radii(self.arr, ree=self.reels, ax=self.ax)

    def tearDown(self):
        plt.close("all")



if __name__ == "__main__":
    unittest.main()
