import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import ternary as pyternary
from pyrolite.plot.tern import ternary
import matplotlib.colors


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
        self.assertEqual(
            type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot
        )

    def test_one(self):
        """Test generation of plot with one record."""
        arr = self.arr[0, :]
        out = ternary(arr)
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(
            type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot
        )

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        arr = self.arr
        out = ternary(arr)
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(
            type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot
        )

    def test_overplotting(self):
        """Test use of the plot for multiple rounds of plotting."""
        arr = self.arr
        out = ternary(arr)
        self.assertTrue(hasattr(out, "tax"))
        out2 = ternary(arr, ax=out)
        self.assertTrue(out.tax is out2.tax) # hasn't added a new ternary axis

    def test_color_tuple(self):
        arr = self.arr
        out = ternary(arr, c=(0.1, 0.2, 0.5, 0.3))
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(
            type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot
        )

    def test_color_hex(self):
        arr = self.arr
        out = ternary(arr, c="#0f0f0f")
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(
            type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot
        )

    def test_color_cmap_c_combination(self):
        """
        Check than array of values specified using `c`
        can be used for a colormap.
        """
        arr = self.arr
        out = ternary(arr, c=np.linspace(0, 10, 10), cmap="viridis")
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(
            type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot
        )

    def test_norm_specified(self):
        arr = self.arr
        out = ternary(
            arr,
            c=np.random.randn(10),
            cmap="viridis",
            norm=matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0),
        )
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(
            type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot
        )

    def test_label_specified(self):
        arr = self.arr
        out = ternary(arr, label="testarr")
        self.assertTrue(hasattr(out, "tax"))
        self.assertEqual(
            type(out.tax), pyternary.ternary_axes_subplot.TernaryAxesSubplot
        )

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
