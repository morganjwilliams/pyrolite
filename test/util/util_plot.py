import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib
import matplotlib.axes
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt

from mpltern.ternary import TernaryAxes
from mpltern.ternary.datasets import get_spiral

from pyrolite.comp.codata import close
from pyrolite.util.plot import *
from pyrolite.util.general import remove_tempdir
from pyrolite.util.skl import ILRTransform, ALRTransform
from pyrolite.util.synthetic import random_composition

try:
    from sklearn.decomposition import PCA

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


class TestMarkerCycle(unittest.TestCase):
    def test_iterable(self):
        mkrs = marker_cycle()
        for i in range(15):
            mkr = next(mkrs)

    def test_makes_line(self):
        mkrs = marker_cycle()
        for i in range(10):
            matplotlib.lines.Line2D([0], [0], marker=next(mkrs))


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


init_axes


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


class TestTernaryTransforms(unittest.TestCase):
    def setUp(self):
        self.ABC = np.array(
            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1 / 3, 1 / 3, 1 / 3], [1, 1, 1]]
        )
        self.xy = np.array([[0, 0], [0.5, 1.0], [1, 0], [0.5, 1 / 3], [0.5, 1 / 3]])

    def test_xy_to_ABC(self):
        out = xy_to_ABC(self.xy)
        self.assertTrue(np.allclose(out, close(self.ABC)))

    def test_ABC_to_xy(self):
        out = ABC_to_xy(self.ABC)
        self.assertTrue(np.allclose(out, self.xy))

    def test_tfm_inversion_xyABC(self):
        out = ABC_to_xy(xy_to_ABC(self.xy))
        self.assertTrue(np.allclose(out, self.xy))

    def test_tfm_inversion_ABCxy(self):
        out = xy_to_ABC(ABC_to_xy(self.ABC))
        self.assertTrue(np.allclose(out, close(self.ABC)))

    def test_xy_to_ABC_yscale(self):
        for yscale in [1.0, 2.0, np.sqrt(3) / 2]:
            out = xy_to_ABC(self.xy, yscale=yscale)
            expect = self.ABC.copy()
            # scale is slightly complicated; will leave for now
            # test inverse
            self.assertTrue(np.allclose(ABC_to_xy(out, yscale=yscale), self.xy))

    def test_ABC_to_xy_yscale(self):
        for yscale in [1.0, 2.0, np.sqrt(3) / 2]:
            out = ABC_to_xy(self.ABC, yscale=yscale)
            expect = self.xy.copy()
            expect[:, 1] *= yscale
            # test scale
            self.assertTrue(np.allclose(out, expect))
            # test inverse

            self.assertTrue(np.allclose(xy_to_ABC(out, yscale=yscale), close(self.ABC)))


class TestTernaryHeatmap(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100, 3)

    def test_default(self):
        out = ternary_heatmap(self.data)
        self.assertTrue(isinstance(out, tuple))
        xe, ye, zi = out
        self.assertTrue(xe.shape == ye.shape)
        # zi could have more or less bins depending on mode..

    def test_aspect(self):
        """
        The ternary heatmap can be used in different aspects for either equilateral
        triangle mode ('eq') or as a triangle which would fit in a unit square ('unit').
        """
        for aspect, expect in [("eq", np.sqrt(3) / 2), ("unit", 1.0)]:
            with self.subTest(aspect=aspect, expect=expect):
                out = ternary_heatmap(self.data, aspect=aspect)
                self.assertTrue(isinstance(out, tuple))
                xe, ye, zi = out
                self.assertTrue(xe.shape == ye.shape)
                ymax = np.nanmax(ye)
                self.assertTrue(ymax < expect)

    def test_histogram(self):
        out = ternary_heatmap(self.data, mode="histogram")
        xe, ye, zi = out
        self.assertTrue(xe.shape == ye.shape)
        self.assertTrue(zi.shape != xe.shape)  # should be higher in x - they're edges
        self.assertTrue(zi.shape == (xe.shape[0] - 1, xe.shape[1] - 1))

    def test_density(self):
        out = ternary_heatmap(self.data, mode="density")
        xe, ye, zi = out
        self.assertTrue(xe.shape == ye.shape)
        self.assertTrue(zi.shape != xe.shape)  # should be higher in x - they're edges
        self.assertTrue(zi.shape == (xe.shape[0] - 1, xe.shape[1] - 1))

    def test_transform(self):
        for tfm, itfm in [
            (alr, inverse_alr),
            (ilr, inverse_ilr),
            (ILRTransform, None),
            (ALRTransform, None),
        ]:
            with self.subTest(tfm=tfm, itfm=itfm):
                out = ternary_heatmap(self.data, transform=tfm, inverse_transform=itfm)
                xe, ye, zi = out

    @unittest.expectedFailure
    def test_need_inverse_transform(self):
        for tfm, itfm in [(alr, None), (ilr, None)]:
            with self.subTest(tfm=tfm, itfm=itfm):
                out = ternary_heatmap(self.data, transform=tfm, inverse_transform=itfm)


class TestBinConversions(unittest.TestCase):
    def setUp(self):
        self.binedges = np.array([0, 1, 2, 3, 4, 5])
        self.bincentres = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        self.asymbinedges = np.array([0, 2, 3, 4, 7])
        self.asymbincentres = np.array([1, 2.5, 3.5, 5.5])

    def test_linear_bin_edges_to_centres(self):
        cs = bin_edges_to_centres(self.binedges)
        self.assertTrue(np.allclose(self.bincentres, cs))

    def test_linear_bin_centres_to_edges(self):
        edgs = bin_centres_to_edges(self.bincentres)
        self.assertTrue(np.allclose(self.binedges, edgs))

    def test_asymmetric_bin_edges_to_centres(self):
        cs = bin_edges_to_centres(self.asymbinedges)
        self.assertTrue(np.allclose(self.asymbincentres, cs))

    @unittest.expectedFailure
    def test_asymmetric_bin_centres_to_edges(self):
        """
        This problem doesn't have a unique solution, only bounds. The simple algorithm
        used can't accurately reconstruct bin edges.
        """
        edgs = bin_centres_to_edges(self.asymbincentres)
        self.assertTrue(np.allclose(self.asymbinedges, edgs))


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
                fig, ax = plt.subplots(1)
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


class TestPercentileContourValuesFromMeshZ(unittest.TestCase):
    def setUp(self):
        x, y = np.mgrid[-1:1:100j, -1:1:100j]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        self.z = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]]).pdf(pos)

    def test_default(self):
        percentile_contour_values_from_meshz(self.z)

    def test_percentiles(self):
        for ps in [[1.0], [0.001], np.linspace(0.001, 1, 10), [0.95, 0.10]]:
            with self.subTest(ps=ps):
                percentile_contour_values_from_meshz(self.z, percentiles=ps)

    def test_resolution(self):
        for res in [10, 100, 1000, 10000]:
            with self.subTest(res=res):
                percentile_contour_values_from_meshz(self.z, resolution=res)


class TestPlotZPercentiles(unittest.TestCase):
    def setUp(self):
        x, y = np.mgrid[-1:1:100j, -1:1:100j]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        self.xi, self.yi = x, y
        self.zi = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]]).pdf(pos)

    def test_default(self):
        plot_Z_percentiles(self.xi, self.yi, self.zi)

    def test_percentiles(self):
        for ps in [[1.0], [0.001], np.linspace(0.001, 1, 10), [0.95, 0.10]]:
            with self.subTest(ps=ps):
                plot_Z_percentiles(self.xi, self.yi, self.zi, percentiles=ps)

    def test_external_ax(self):
        fig, ax = plt.subplots(1)
        plot_Z_percentiles(self.xi, self.yi, self.zi, ax=ax)

    def test_extent(self):
        for extent in [[-1, 1, -1, 1], [-0.01, 0.99, -1.01, -0.01], [-2, 2, -2, -2]]:
            with self.subTest(extent=extent):
                plot_Z_percentiles(self.xi, self.yi, self.zi, extent=extent)

    def tearDown(self):
        plt.close("all")


class TestNaNScatter(unittest.TestCase):
    """
    Tests the nan_scatter utility plotting function.
    """

    def setUp(self):
        self.x = np.random.randn(1000) - 1
        self.y = 2 + np.random.randn(1000)
        self.x[self.x < -1] = np.nan
        self.y[self.y < 2] = np.nan

    def test_plot(self):
        fig, ax = plt.subplots()
        ax = nan_scatter(self.x, self.y, ax=ax)
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))

    def tearDown(self):
        plt.close("all")


class TestSaveUtilities(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots(1)

    def test_get_full_extent(self):
        extent = get_full_extent(self.ax)

    def tearDown(self):
        plt.close("all")


class TestSaveFunctions(unittest.TestCase):
    """
    Tests the collection of plot saving functions.
    """

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.tempdir = Path("./testing_temp_figures")
        if not self.tempdir.exists():
            self.tempdir.mkdir()

    def test_save_figure(self):
        save_figure(self.fig, name="test_fig", save_at=str(self.tempdir))

    def test_save_axes(self):
        save_axes(self.ax, name="test_ax", save_at=str(self.tempdir))

    def tearDown(self):
        remove_tempdir(str(self.tempdir))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
