import unittest

import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyrolite.util.classification import USDASoilTexture
from pyrolite.util.plot.style import (
    _mpl_sp_kw_split,
    color_ternary_polygons_by_centroid,
    linekwargs,
    mappable_from_values,
    marker_cycle,
    patchkwargs,
    scatterkwargs,
    ternary_color,
)
from pyrolite.util.synthetic import normal_frame


class TestMarkerCycle(unittest.TestCase):
    def test_iterable(self):
        mkrs = marker_cycle()
        for i in range(15):
            mkr = next(mkrs)

    def test_line_compatible(self):
        mkrs = marker_cycle()
        for i in range(10):
            matplotlib.lines.Line2D([0], [0], marker=next(mkrs))


class TestSplitKwargs(unittest.TestCase):
    """
    Test the function used to split scatter and line
    specific keyword arguments (WIP).
    """

    def test_default(self):
        kw = dict(c="k", color="r", linewidth=0.5)
        sctr_kw, line_kw = _mpl_sp_kw_split(kw)
        self.assertTrue("c" in sctr_kw)
        self.assertTrue("c" not in line_kw)
        self.assertTrue("color" not in sctr_kw)
        self.assertTrue("color" in line_kw)
        self.assertTrue("linewidth" not in sctr_kw)
        self.assertTrue("linewidth" in line_kw)


class TestMappableFromValues(unittest.TestCase):

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)

    def test_colorbar(self):
        for src in [np.random.randn(10), pd.Series(np.random.randn(10))]:
            with self.subTest(src=src):
                mappable = mappable_from_values(src)
                plt.colorbar(mappable, ax=self.ax)

    def tearDown(self):
        plt.close("all")


class TestTernaryColor(unittest.TestCase):
    def setUp(self):
        self.df = normal_frame(
            columns=["CaO", "MgO", "FeO"],
            size=100,
            seed=42,
            cov=np.array([[0.8, 0.3], [0.3, 0.8]]),
        )

    def test_ternary_color(self):
        for src in [self.df, self.df.values]:
            with self.subTest(src=src):
                colors = ternary_color(
                    src,
                    alpha=0.9,
                    colors=["green", "0.5", [0.9, 0.1, 0.5, 0.9]],
                    coefficients=[0.5, 0.5, 2],
                )
                ax = self.df.pyroplot.scatter(c=colors)

    def tearDown(self):
        plt.close("all")


class TestTernaryPolygonCentroidColors(unittest.TestCase):
    def setUp(self):
        self.clf = USDASoilTexture()
        self.ax = self.clf.add_to_axes()

    def test_ternary_centroid_color(self):
        color_ternary_polygons_by_centroid(self.ax)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
