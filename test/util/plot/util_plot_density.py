import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pyrolite.util.plot.density import (
    percentile_contour_values_from_meshz,
    plot_Z_percentiles,
)
from pyrolite.util.plot.legend import proxy_line
from matplotlib.lines import _get_dash_pattern, _scale_dashes
import matplotlib.colors


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

    def test_specified_contour_labels(self):
        contour_labels = ["95th", "66th", "33rd"]
        cs = plot_Z_percentiles(
            self.xi, self.yi, zi=self.zi, contour_labels=contour_labels
        )
        for contour_label, label in zip(contour_labels, cs.labelTextsList):
            label = label.get_text()
            self.assertTrue(contour_label == label)

    def test_styling_specified(self):
        fig, ax = plt.subplots(1)

        colors = [matplotlib.colors.to_rgba(c) for c in ["g", "b", "k"]]
        linestyles = [_get_dash_pattern(d) for d in ["-", "--", "-."]]
        linewidths = [1, 2, 3]
        cs = plot_Z_percentiles(
            self.xi,
            self.yi,
            zi=self.zi,
            ax=ax,
            percentiles=[0.95, 0.66, 0.33],
            colors=colors,
            linestyles=linestyles,
            linewidths=linewidths,
        )
        for contour, color, ls, lw in zip(
            cs.collections, colors, linestyles, linewidths
        ):
            self.assertTrue((contour.get_color() == color).all())
            self.assertEqual(contour.get_linestyle(), [_scale_dashes(*ls, lw)])
            self.assertEqual(contour.get_linewidth(), lw)

    def test_linestyles_specified(self):
        plot_Z_percentiles(
            self.xi,
            self.yi,
            zi=self.zi,
            percentiles=[0.95, 0.66, 0.33],
        )

    def test_percentiles(self):
        for ps in [[1.0], [0.01], np.linspace(0.001, 1, 10), [0.95, 0.10]]:
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
