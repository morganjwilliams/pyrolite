import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyrolite.geochem import REE
from pyrolite.plot.spider import spider, REE_v_radii
import logging

logger = logging.getLogger(__name__)


class TestSpiderplot(unittest.TestCase):
    """Tests the Spiderplot functionality."""

    def setUp(self):
        self.els = REE()
        self.df = test_df(self.els)

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
        out = spider(self.df.values, plot=False, fill=False)
        self.assertTrue(isinstance(out, Maxes.Axes))
        plt.close("all")

    def test_valid_style(self):
        """Test valid styling options."""
        pass

    def test_log_on_irrellevant_style_options(self):
        """Test stability under additional kwargs."""
        style = {"thingwhichisnotacolor": "notacolor", "irrelevant": "red"}
        with self.assertLogs(level="INFO") as cm:
            # with self.assertWarns(UserWarning):
            ax = spider(self.df.values, **style)

        plt.close("all")

    @unittest.expectedFailure
    def test_invalid_style_options(self):
        """Test stability under invalid style values."""
        style = {"color": "notacolor", "marker": "red"}
        spider(self.df.values, **style)
        plt.close("all")

    def test_pandas_flavour(self):
        """Test generation of plot using the pandas flavour registered method."""
        self.df.pyroplot.spider()

    def tearDown(self):
        plt.close("all")


class TestREERadiiPlot(unittest.TestCase):
    """Tests the REE_radii_plot functionality."""

    def setUp(self):
        self.reels = REE(output="string")
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.reels, np.random.rand(len(reels), 10))}
        )

    def test_none(self):
        """Test generation of plot with no data."""
        ax = REE_v_radii()

    def test_one_np(self):
        ax = REE_v_radii(self.df.values, ree=self.reels)

    def test_one_pandas(self):
        ax = self.df.pyroplot.REE(ree=self.reels)

    def tearDown(self):
        plt.close("all")



if __name__ == "__main__":
    unittest.main()
