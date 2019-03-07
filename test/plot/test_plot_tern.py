import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import ternary as pyternary
from pyrolite.plot.tern import ternary

logger = logging.getLogger(__name__)

class TestTernaryplot(unittest.TestCase):
    """Tests the Ternary functionality."""

    def setUp(self):
        self.arr = np.random.rand(10, 3)

    def test_none(self):
        """Test generation of plot with no data."""
        arr = np.empty(0)
        out = ternary(arr)
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot)
        plt.close("all")

    def test_one(self):
        """Test generation of plot with one record."""
        arr = self.arr[0, :]
        out = ternary(arr)
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot)
        plt.close("all")

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        arr = self.arr
        out = ternary(arr)
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
