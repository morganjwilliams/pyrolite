import unittest

import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np

from pyrolite.util.plot.interpolation import (
    get_contour_paths,
    interpolate_path,
    interpolated_patch_path,
)


class TestInterpolatePath(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.ax.plot([0, 1, 2, 3, 0], [0, 0.1, 1, 2, 0])  # a closed path
        self.path = self.ax.lines[0].get_path()

    def test_interpolate(self):
        for aspath in (True, False):
            for closefirst in (True, False):
                with self.subTest(aspath=aspath, closefirst=closefirst):
                    interp_path = interpolate_path(
                        self.path, aspath=aspath, closefirst=closefirst
                    )

    def tearDown(self):
        plt.close("all")


class InterpolatedPathPatch(unittest.TestCase):
    """
    Tests the interpolated_path_patch utility function.
    """

    def setUp(self):
        self.patch = matplotlib.patches.Ellipse((0, 0), 1, 2)

    def test_default(self):
        path = interpolated_patch_path(self.patch)
        self.assertTrue(isinstance(path, matplotlib.path.Path))

    def test_resolution(self):
        for res in [2, 10, 100]:
            with self.subTest(res=res):
                path = interpolated_patch_path(self.patch, resolution=res)
                self.assertTrue(path.vertices.shape[0] == res)


class TestContourPaths(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        # this is the contour demo data from matplotlib, for refrence
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-2.0, 2.0, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-(X**2) - Y**2)
        Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
        Z = (Z1 - Z2) * 2
        self.contours = self.ax.contour(X, Y, Z)
        self.clabels = self.ax.clabel(self.contours, inline=True)

    def test_default(self):
        # can pass either axes (with nothing else on it.. ) or the contourset
        for src in [self.ax, self.contours]:
            with self.subTest(src=src):
                paths, names, styles = get_contour_paths(src)
                self.assertTrue(len(set([len(paths), len(names), len(styles)])) == 1)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
