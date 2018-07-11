import unittest
import pandas as pd
import numpy as np
import matplotlib.axes as matax
import matplotlib.pyplot as plt
from pyrolite.util.plot import *
from sklearn.decomposition import PCA
from pyrolite.compositions import close


# add_colorbar

# ABC_to_tern_xy

# tern_heatmapcoords

# proxy_rect

# proxy_line

# draw_vector

# vector_to_line

class TestDrawVector(unittest.TestCase):

    def setUp(self):
        xs = 1./(np.random.randn(5)+4)
        self.X = np.array([xs, 1-xs])
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
        plt.close('all')


class TestVectorToLine(unittest.TestCase):

    def setUp(self):
        xs = 1./(np.random.randn(5)+4)
        self.X = np.array([xs, 1-xs])
        self.X = close(self.X)

    def test_to_line(self):
        pca = PCA(n_components=2)
        d = self.X
        pca.fit(d)
        for variance, vector in zip(pca.explained_variance_, pca.components_):
            line = vector_to_line(pca.mean_[:2], vector[:2], variance, spans=6)

        self.assertTrue(isinstance(line, np.ndarray))
        self.assertTrue(line.shape[1] == 2)

# plot_2dhull


class TestNaNScatter(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(1000) - 1
        self.y = 2 + np.random.randn(1000)
        self.x[self.x < -1] = np.nan
        self.y[self.y < 2] = np.nan

    def test_plot(self):
        fig, ax = plt.subplots()
        ax = nan_scatter(ax, self.x, self.y)
        self.assertTrue(isinstance(ax, matax.Axes))

    def tearDown(self):
        plt.close('all')


# save_figure

# save_axes

# get_full_extent


if __name__ == '__main__':
    unittest.main()
