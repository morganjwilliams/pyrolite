import unittest
import matplotlib.axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyrolite.plot.stem import stem

logger = logging.getLogger(__name__)

class TestStem(unittest.TestCase):
    """Tests the Stem plot functionality."""

    def setUp(self):
        self.x = np.linspace(0, 10, 10) + np.random.randn(10) / 2.0
        self.y = np.random.rand(10)

    def test_default(self):
        ax = stem(self.x, self.y)
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))
        ax.invert_yaxis()

    def test_orientation(self):
        for orientation in ['horizontal', 'vertical', 'h', 'v']:
            with self.subTest(orientation=orientation):
                ax = stem(self.x, self.y, orientation=orientation)

    def test_axis_specified(self):
        """Test generation of plot with axis specified."""
        fig, ax = plt.subplots(1)
        ax2 = stem(self.x, self.y, ax=ax)
        self.assertTrue(isinstance(ax2, matplotlib.axes.Axes))

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
