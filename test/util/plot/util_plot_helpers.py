import unittest
import numpy as np
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
from pyrolite.comp.codata import close
from pyrolite.util.synthetic import random_composition
from pyrolite.util.skl.transform import ILRTransform, ALRTransform
from pyrolite.util.plot.helpers import (
    plot_2dhull,
    plot_cooccurence,
    plot_pca_vectors,
    plot_stdev_ellipses,
    draw_vector,
    vector_to_line,
    nan_scatter,
    rect_from_centre
)

try:
    from sklearn.decomposition import PCA

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


class TestPlotCooccurence(unittest.TestCase):
    def setUp(self):
        self.rdata = pd.DataFrame(
            random_composition(size=200, D=4, missing="MCAR"),
            columns=["MgO", "SiO2", "CaO", "TiO2"],
        )

    def test_default(self):
        ax = plot_cooccurence(self.rdata)

    def test_normalize(self):
        for normalize in [True, False]:
            with self.subTest(normalize=normalize):
                ax = plot_cooccurence(self.rdata, normalize=normalize)

    def test_log(self):
        for log in [True, False]:
            with self.subTest(log=log):
                ax = plot_cooccurence(self.rdata, log=log)

    def test_colorbar(self):
        for colorbar in [True, False]:
            with self.subTest(colorbar=colorbar):
                ax = plot_cooccurence(self.rdata, colorbar=colorbar)

    def test_external_ax(self):
        fig, ax = plt.subplots(1)
        ax = plot_cooccurence(self.rdata, ax=ax)


class TestPlotStDevEllipses(unittest.TestCase):
    def setUp(self):

        self.comp3d = random_composition(size=100, D=3)
        self.T = ILRTransform()
        self.comp2d = self.T.transform(self.comp3d)

    def test_default(self):
        for comp in [self.comp2d]:
            with self.subTest(comp=comp):
                plot_stdev_ellipses(comp, transform=self.T.inverse_transform)

    def test_axis_specified(self):
        for comp in [self.comp2d]:
            with self.subTest(comp=comp):
                fig, ax = plt.subplots(1, subplot_kw=dict(projection="ternary"))
                plot_stdev_ellipses(comp, ax=ax, transform=self.T.inverse_transform)

    def test_transform(self):
        for tfm in [None, ILRTransform, ALRTransform]:
            with self.subTest(tfm=tfm):
                if callable(tfm):
                    T = tfm()
                    comp = T.transform(self.comp3d)
                    transform = T.inverse_transform
                else:
                    transform = None
                    comp = self.comp2d

                plot_stdev_ellipses(comp, transform=transform)

    def tearDown(self):
        plt.close("all")


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestPlotPCAVectors(unittest.TestCase):
    def setUp(self):
        self.comp3d = random_composition(size=100, D=3)
        self.T = ILRTransform()
        self.comp2d = self.T.transform(self.comp3d)

    def test_default(self):
        for comp in [self.comp2d, self.comp3d]:
            with self.subTest(comp=comp):
                plot_pca_vectors(comp)

    def test_axis_specified(self):
        for comp in [self.comp2d, self.comp3d]:
            with self.subTest(comp=comp):
                fig, ax = plt.subplots()
                plot_pca_vectors(comp, ax=ax)

    def test_transform(self):
        for tfm in [None, ILRTransform, ALRTransform]:
            with self.subTest(tfm=tfm):
                if callable(tfm):
                    T = tfm()
                    comp = T.transform(self.comp3d)
                    transform = T.inverse_transform
                else:
                    transform = None
                    comp = self.comp2d

                plot_pca_vectors(comp, transform=transform)

    def tearDown(self):
        plt.close("all")


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestDrawVector(unittest.TestCase):
    """
    Tests the draw_vector utility function.
    """

    def setUp(self):
        xs = 1.0 / (np.random.randn(5) + 4)
        self.X = np.array([xs, 1 - xs])
        self.X = close(self.X)

    def test_plot(self):
        fig, ax = plt.subplots(1)
        pca = PCA(n_components=2)
        d = self.X
        pca.fit(d)
        for variance, vector in zip(pca.explained_variance_, pca.components_):
            v = vector[:2] * 3 * np.sqrt(variance)
            draw_vector(pca.mean_[:2], pca.mean_[:2] + v, ax=ax)

    def tearDown(self):
        plt.close("all")


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestVectorToLine(unittest.TestCase):
    """
    Tests the vector_to_line utility function.
    """

    def setUp(self):
        xs = 1.0 / (np.random.randn(5) + 4)
        self.X = np.array([xs, 1 - xs])
        self.X = close(self.X)

    def test_to_line(self):
        pca = PCA(n_components=2)
        d = self.X
        pca.fit(d)
        for variance, vector in zip(pca.explained_variance_, pca.components_):
            line = vector_to_line(pca.mean_[:2], vector[:2], variance, spans=6)

        self.assertTrue(isinstance(line, np.ndarray))
        self.assertTrue(line.shape[1] == 2)


class Test2DHull(unittest.TestCase):
    """
    Tests the plot_2dhull utility function.
    """

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.data = np.random.random((2, 10)).T

    def test_2d_hull(self):
        lines = plot_2dhull(self.data, ax=self.ax)
        self.assertTrue(isinstance(lines[0], matplotlib.lines.Line2D))

    def test_2d_hull_splines(self):
        lines = plot_2dhull(self.data, ax=self.ax, splines=True)
        self.assertTrue(isinstance(lines[0], matplotlib.lines.Line2D))

    def tearDown(self):
        plt.close("all")


class TestRectFromCentre(unittest.TestCase):
    def setUp(self):
        self.xy = (0.5, 1)

    def test_default(self):
        rect = rect_from_centre(*self.xy, dx=0.5, dy=1.0)
        self.assertIsInstance(rect, matplotlib.patches.Rectangle)


class TestNaNScatter(unittest.TestCase):
    def setUp(self):
        # some x-y data with nans
        self.xs, self.ys = np.random.randn(20), np.random.randn(20)
        self.xs[self.xs < -0.5] = np.nan
        self.ys[self.ys < -0.8] = np.nan

    def test_default(self):
        ax = nan_scatter(self.xs, self.ys)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_secondary_plotting(self):
        """Test re-plotting on exsiting axes with a divider"""
        ax = nan_scatter(self.xs[:10], self.ys[:10])
        ax = nan_scatter(self.xs[10:], self.ys[10:], ax=ax)
        self.assertIsInstance(ax, matplotlib.axes.Axes)


if __name__ == "__main__":
    unittest.main()
