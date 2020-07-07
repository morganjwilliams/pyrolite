import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpltern.ternary import TernaryAxes
from mpltern.ternary.datasets import get_spiral
from pyrolite.util.plot.axes import (
    replace_with_ternary_axis,
    axes_to_ternary,
    get_axes_index,
    get_ordered_axes,
    get_twins,
    share_axes,
    add_colorbar,
)


class TestReplaceWithTernaryAxis(unittest.TestCase):
    def test_default(self):
        ix = 1
        fig, ax = plt.subplots(1, 2)
        tax = replace_with_ternary_axis(ax[ix])
        self.assertTrue(hasattr(fig, "orderedaxes"))
        self.assertEqual(fig.orderedaxes[ix], tax)
        self.assertIsInstance(tax, TernaryAxes)


class TestAxesToTernary(unittest.TestCase):
    def setUp(self):
        self.tlr = get_spiral()

    def test_default(self):
        ix = 1
        fig, ax = plt.subplots(1, 2)
        ax = axes_to_ternary(ax[ix])
        self.assertIsInstance(ax, list)
        self.assertIsInstance(ax[ix], TernaryAxes)

    def test_multiple_grid(self):
        ix = [1, 3]
        fig, ax = plt.subplots(2, 2)
        ax = ax.flat
        ax = axes_to_ternary([ax[i] for i in ix])
        self.assertIsInstance(ax, list)
        for i in ix:
            self.assertIsInstance(ax[i], TernaryAxes)

    def test_plot(self):
        ix = 1
        fig, ax = plt.subplots(1, 2)
        ax = axes_to_ternary(ax[ix])
        ax[ix].plot(*self.tlr, "k")


class TestSetAxesToTernary(unittest.TestCase):
    def setUp(self):
        self.tlr = get_spiral()

    def test_default(self):
        ix = 1
        fig, ax = plt.subplots(1, 2)
        ax = axes_to_ternary(ax[ix])
        self.assertTrue(hasattr(fig, "orderedaxes"))
        self.assertIsInstance(fig.orderedaxes[ix], TernaryAxes)

    def test_multiple_grid(self):
        ix = [1, 3]
        fig, ax = plt.subplots(2, 2)
        ax = ax.flat
        ax = axes_to_ternary([ax[i] for i in ix])
        self.assertTrue(hasattr(fig, "orderedaxes"))
        for i in ix:
            self.assertIsInstance(fig.orderedaxes[i], TernaryAxes)

    def test_plot(self):
        ix = 1
        fig, ax = plt.subplots(1, 2)
        ax = axes_to_ternary(ax[ix])
        ax[ix].plot(*self.tlr, "k")


class GetAxesIndex(unittest.TestCase):
    def test_default(self):
        ix = 2
        grid = (5, 1)
        fig, ax = plt.subplots(*grid)
        ax = ax.flat
        triple = get_axes_index(ax[ix])
        self.assertEqual(triple, (*grid, ix + 1))

    def test_grid(self):
        ix = 2
        grid = (2, 3)
        fig, ax = plt.subplots(*grid)
        ax = ax.flat
        triple = get_axes_index(ax[ix])
        self.assertEqual(triple, (*grid, ix + 1))

    def tearDown(self):
        plt.close("all")


class TestShareAxes(unittest.TestCase):
    """
    Tests the share_axes utility function.
    """

    def test_default(self):
        fig, ax = plt.subplots(5)
        share_axes(ax)
        for a in ax:
            self.assertTrue(
                all([i in a.get_shared_x_axes().get_siblings(a) for i in ax])
            )
            self.assertTrue(
                all([i in a.get_shared_y_axes().get_siblings(a) for i in ax])
            )

    def test_which(self):
        for which, methods in [
            ("xy", ["get_shared_x_axes", "get_shared_y_axes"]),
            ("x", ["get_shared_x_axes"]),
            ("y", ["get_shared_y_axes"]),
            ("both", ["get_shared_x_axes", "get_shared_y_axes"]),
        ]:
            with self.subTest(which=which, methods=methods):
                fig, ax = plt.subplots(5)
                share_axes(ax, which=which)
                for a in ax:
                    for m in methods:
                        self.assertTrue(
                            all([i in getattr(a, m)().get_siblings(a) for i in ax])
                        )
                plt.close("all")

    def tearDown(self):
        plt.close("all")


class TestGetTwinAxes(unittest.TestCase):
    def setUp(self):
        fig, ax = plt.subplots(1)
        self.ax = ax
        self.axy = self.ax.twiny()  # independent x axis on top, same y axis
        self.axx = self.ax.twinx()  # independent y axis on right, same x axis

    def test_y(self):
        out = get_twins(self.ax, which="y")
        self.assertNotIn(self.axx, out)
        self.assertIn(self.axy, out)

    def test_x(self):
        out = get_twins(self.ax, which="x")
        self.assertNotIn(self.axy, out)
        self.assertIn(self.axx, out)

    def test_both(self):
        out = get_twins(self.ax, which="xy")
        self.assertIn(self.axx, out)
        self.assertIn(self.axy, out)
        self.assertTrue(len(out) == 2)


class TestAddColorbar(unittest.TestCase):
    """
    Tests the add_colorbar utility function.
    """

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.mappable = plt.imshow(np.random.random((10, 10)), cmap=plt.cm.BuPu_r)

    def test_colorbar(self):
        add_colorbar(self.mappable)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
