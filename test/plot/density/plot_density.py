import unittest
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
import logging
from pyrolite.plot.density import density
from pyrolite.plot.density.ternary import ternary_heatmap
from pyrolite.comp.codata import close, ILR, ALR, inverse_ILR, inverse_ALR
from pyrolite.util.skl.transform import ILRTransform, ALRTransform

logger = logging.getLogger(__name__)


class TestDensityplot(unittest.TestCase):
    """Tests the Densityplot functionality."""

    def setUp(self):
        self.cols = ["MgO", "SiO2", "CaO"]
        data = np.array([3, 5, 7])
        cov = np.array([[2, -1, -0.5], [-1, 2, -1], [-0.5, -1, 2]])

        self.biarr = multivariate_normal(data[:2], cov[:2, :2], 100)
        self.triarr = np.abs(multivariate_normal(data, cov, 100))  # needs to be >0

        self.biarr[0, 1] = np.nan
        self.triarr[0, 1] = np.nan

    def test_none(self):
        """Test generation of plot with no data."""
        for arr in [np.empty(0)]:
            with self.subTest(arr=arr):
                out = density(arr)
                self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                plt.close("all")

    def test_one(self):
        """Test generation of plot with one record."""

        for arr in [self.biarr[0, :], self.triarr[0, :]]:
            with self.subTest(arr=arr):
                out = density(arr)
                self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                plt.close("all")

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        for arr in [self.biarr, self.triarr]:
            with self.subTest(arr=arr):
                out = density(arr, vmin=0)
                self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                plt.close("all")

    def test_modes(self):  #
        """Tests different ploting modes."""
        for arr in [self.biarr, self.triarr]:
            with self.subTest(arr=arr):
                for mode in ["density", "hist2d", "hexbin"]:
                    with self.subTest(mode=mode):
                        try:
                            out = density(arr, mode=mode, vmin=0)
                            self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                            plt.close("all")
                        except NotImplementedError:  # some are not implemented for 3D
                            pass

    def test_bivariate_logscale(self):  #
        """Tests logscale for different ploting modes using bivariate data."""
        arr = self.biarr
        with np.errstate(invalid="ignore"):  # ignore for tests
            arr[arr < 0] = np.nan
        for logspacing in [(True, True), (False, False), (False, True), (True, False)]:
            lx, ly = logspacing
            with self.subTest(logx=lx, logy=ly):
                for mode in ["density", "hist2d", "hexbin"]:
                    with self.subTest(mode=mode):
                        out = density(arr, mode=mode, logx=lx, logy=ly)
                        self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                    plt.close("all")

    def test_colorbar(self):
        for arr in [self.biarr, self.triarr]:
            for mode in ["density", "hist2d"]: # hexbin won't work for triarr
                with self.subTest(mode=mode):
                    out = density(arr, mode=mode, colorbar=True)
                    self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                plt.close("all")

    def test_contours(self):
        contours = [0.9, 0.5]
        for arr in [self.biarr, self.triarr]:
            for mode in ["density"]:
                with self.subTest(mode=mode, arr=arr):
                    out = density(arr, mode=mode, contours=contours)
                    self.assertTrue(isinstance(out, matplotlib.axes.Axes))
                plt.close("all")

    def test_contours_levels(self):
        arr = self.biarr
        mode = "density"
        for levels in [3, None]:
            with self.subTest(levels=levels):
                out = density(arr, mode=mode, contours=True, percentiles=False)
                self.assertTrue(isinstance(out, matplotlib.axes.Axes))
            plt.close("all")

    def test_cmap(self):
        arr = self.biarr
        cmap = "viridis"
        for mode in ["density", "hist2d", "hexbin"]:
            with self.subTest(mode=mode):
                out = density(arr, mode=mode, cmap=cmap)
                self.assertTrue(isinstance(out, matplotlib.axes.Axes))
            plt.close("all")

    def tearDown(self):
        plt.close("all")


class TestTernaryHeatmap(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100, 3)

    def test_default(self):
        out = ternary_heatmap(self.data)
        self.assertTrue(isinstance(out, tuple))
        coords, H, data = out
        self.assertTrue(coords[0].shape == coords[1].shape)
        # zi could have more or less bins depending on mode..

    def test_histogram(self):
        out = ternary_heatmap(self.data, mode="histogram")
        xe, ye, zi = out
        coords, H, data = out
        self.assertTrue(coords[0].shape == coords[1].shape)

    def test_density(self):
        out = ternary_heatmap(self.data, mode="density")
        coords, H, data = out
        self.assertTrue(coords[0].shape == coords[1].shape)

    def test_transform(self):
        for tfm, itfm in [
            (ALR, inverse_ALR),
            (ILR, inverse_ILR),
            (ILRTransform, None),
            (ALRTransform, None),
        ]:
            with self.subTest(tfm=tfm, itfm=itfm):
                out = ternary_heatmap(self.data, transform=tfm, inverse_transform=itfm)
                coords, H, data = out

    @unittest.expectedFailure
    def test_need_inverse_transform(self):
        for tfm, itfm in [(ALR, None), (ILR, None)]:
            with self.subTest(tfm=tfm, itfm=itfm):
                out = ternary_heatmap(self.data, transform=tfm, inverse_transform=itfm)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
