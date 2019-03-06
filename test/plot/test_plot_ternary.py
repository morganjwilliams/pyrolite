import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import ternary as pyternary
from pyrolite.plot.ternary import ternary
from pyrolite.util.pd import test_df

logger = logging.getLogger(__name__)


class TestTernaryplot(unittest.TestCase):
    """Tests the Ternary functionality."""

    def setUp(self):
        self.cols = ["MgO", "CaO", "SiO2"]
        self.df = test_df(self.cols)

    def test_none(self):
        """Test generation of plot with no data."""
        df = pd.DataFrame(columns=self.cols)
        out = ternary(df)
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot)
        plt.close("all")

    def test_one(self):
        """Test generation of plot with one record."""
        df = self.df.head(1)
        out = ternary.ternary(df)
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot)
        plt.close("all")

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        df = self.df.loc[:, :]
        out = ternary.ternary(df)
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot)
        plt.close("all")

    def test_overplotting(self):
        """Test use of the plot for multiple rounds of plotting."""
        pass

    def tearDown(self):
        plt.close("all")



if __name__ == "__main__":
    unittest.main()
