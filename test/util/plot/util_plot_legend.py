import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines
from pyrolite.util.plot.legend import proxy_line, proxy_rect, modify_legend_handles


class TestLegendProxies(unittest.TestCase):
    """
    Tests the proxy_rect and proxy_line utility functions.
    """

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)

    def test_proxy_rect(self):
        rect = proxy_rect()
        self.assertTrue(isinstance(rect, matplotlib.patches.Polygon))
        self.assertTrue(isinstance(rect, matplotlib.patches.Rectangle))

    def test_proxy_rect(self):
        line = proxy_line()
        self.assertTrue(isinstance(line, matplotlib.lines.Line2D))

    def tearDown(self):
        plt.close("all")


class TestModifyLegendHandles(unittest.TestCase):
    """
    Tests the modify_legend_handles utility function.
    """

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.ax.plot(np.random.random(10), np.random.random(10), color="g", label="a")

    def test_modify_legend_handles(self):
        _hndls, labls = modify_legend_handles(self.ax, **{"color": "k"})
        self.assertTrue(_hndls[0].get_color() == "k")

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
