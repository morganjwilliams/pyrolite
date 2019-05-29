import unittest
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
from pyrolite.geochem import REE
from pyrolite.plot.spider import spider, REE_v_radii
from pyrolite.util.synthetic import random_composition
import logging

logger = logging.getLogger(__name__)


try:
    import statsmodels.api as sm

    HAVE_SM = True
except ImportError:
    HAVE_SM = False


class TestSpiderplot(unittest.TestCase):
    """Tests the Spiderplot functionality."""

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.els = REE()
        self.arr = random_composition(size=10, D=len(self.els))

    def test_none(self):
        """Test generation of plot with no data."""
        ax = spider(np.empty(0))
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

    def test_one(self):
        """Test generation of plot with one record."""
        ax = spider(self.arr[0, :])
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        ax = spider(self.arr)
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

    def test_axis_specified(self):
        """Test generation of plot with axis specified."""
        ax = spider(self.arr, ax=self.ax)
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

    def test_modes(self):
        """Test all mode functionality is available."""
        for mode in ["plot", "fill", "binkde", "kde", "hist"]:
            with self.subTest(mode=mode):
                ax = spider(self.arr, mode=mode)

    @unittest.skipUnless(HAVE_SM, "Requires statsmodels")
    def test_mode_ckde(self):
        for mode in ["ckde"]:
            with self.subTest(mode=mode):
                ax = spider(self.arr, mode=mode)

    def test_invalid_mode_raises_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            for arr in [self.arr]:
                ax = spider(arr, mode="notamode")

    def test_valid_style(self):
        """Test valid styling options."""
        for sty in [{"c": "k"}]:
            ax = spider(self.arr, **sty)
            self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

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

    def test_index(self):
        for index in ["radii", "elements"]:
            with self.subTest(index=index):
                ax = REE_v_radii(self.arr, ree=self.reels, index=index)

    def test_modes(self):
        """Test all mode functionality is available."""
        for mode in ["plot", "fill", "binkde", "kde", "hist"]:
            with self.subTest(mode=mode):
                ax = REE_v_radii(self.arr, ree=self.reels, mode=mode)

    @unittest.skipUnless(HAVE_SM, "Requires statsmodels")
    def test_mode_ckde(self):
        for mode in ["ckde"]:
            with self.subTest(mode=mode):
                ax = REE_v_radii(self.arr, ree=self.reels, mode=mode)

    def test_external_ax(self):
        ax = REE_v_radii(self.arr, ree=self.reels, ax=self.ax)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
