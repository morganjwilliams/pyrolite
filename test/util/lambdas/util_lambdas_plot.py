import unittest
import numpy as np
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt

from pyrolite.util.lambdas.plot import (
    plot_lambdas_components,
    plot_tetrads_components,
    plot_profiles,
)


class TestPlotLambdasComponents(unittest.TestCase):
    def setUp(self):
        self.lambdas = np.array([0.1, 1.0, 10.0, 100.0])

    def test_default(self):
        ax = plot_lambdas_components(self.lambdas)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def tearDown(self):
        plt.close("all")


class TestPlotTetradsComponents(unittest.TestCase):
    def setUp(self):
        self.taus = np.array([-1, 1.0, -0.5, 1.5])

    def test_default(self):
        ax = plot_tetrads_components(self.taus)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def tearDown(self):
        plt.close("all")


class TestPlotProfiles(unittest.TestCase):
    def setUp(self):
        self.lambdas = np.array([[0.1, 1.0, 10.0, 100.0], [0.1, 1.0, 10.0, 100.0]])
        self.lambdataus = np.array(
            [
                [0.1, 1.0, 10.0, 100.0, -1, 1.0, -0.5, 1.5],
                [0.1, 1.0, 10.0, 100.0, -1, 1.0, -0.5, 1.5],
            ]
        )

    def test_default(self):
        ax = plot_profiles(self.lambdas)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_lambdas_and_taus(self):
        ax = plot_profiles(self.lambdataus, tetrads=True)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
