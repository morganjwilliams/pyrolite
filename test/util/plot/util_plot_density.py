import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pyrolite.util.plot.density import (
    percentile_contour_values_from_meshz,
    plot_Z_percentiles,
)


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
                pc, cs = percentile_contour_values_from_meshz(self.z, percentiles=ps)

    def test_resolution(self):
        for res in [10, 100, 1000, 10000]:
            with self.subTest(res=res):
                pc, cs = percentile_contour_values_from_meshz(self.z, resolution=res)

    def test_ask_below_minimum(self):
        for ps in [[0.0001], [0.000001]]:
            with self.subTest(ps=ps):
                pc, cs = percentile_contour_values_from_meshz(
                    self.z, percentiles=ps, resolution=5
                )
                self.assertIn("min", pc)


class TestPlotZPercentiles(unittest.TestCase):
    def setUp(self):
        x, y = np.mgrid[-1:1:100j, -1:1:100j]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        self.xi, self.yi = x, y
        self.zi = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]]).pdf(pos)

    def test_default(self):
        plot_Z_percentiles(self.xi, self.yi, zi=self.zi)

    def test_percentiles(self):
        for ps in [[1.0], [0.001], np.linspace(0.001, 1, 10), [0.95, 0.10]]:
            with self.subTest(ps=ps):
                plot_Z_percentiles(self.xi, self.yi, zi=self.zi, percentiles=ps)

    def test_external_ax(self):
        fig, ax = plt.subplots(1)
        plot_Z_percentiles(self.xi, self.yi, zi=self.zi, ax=ax)

    def test_extent(self):
        for extent in [[-1, 1, -1, 1], [-0.01, 0.99, -1.01, -0.01], [-2, 2, -2, -2]]:
            with self.subTest(extent=extent):
                plot_Z_percentiles(self.xi, self.yi, zi=self.zi, extent=extent)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
