import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes

from pyrolite.plot.parallel import parallel

import pyrolite.data.Aitchison


class TestParallelCoordinatePlot(unittest.TestCase):
    """Tests the plot.parallel.parallel functionality."""

    def setUp(self):
        self.df = pyrolite.data.Aitchison.load_coxite()
        self.comp = [i for i in self.df.columns if i != "Depth"]

    def test_default(self):
        ax = parallel(self.df)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_colorby_column(self):
        ax = parallel(self.df, color_by="Depth")
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_colorby_array(self):
        ax = parallel(self.df, color_by=self.df.Depth.values)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_colorby_list(self):
        ax = parallel(self.df, color_by=self.df.Depth.values.tolist())
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_legend_switch(self):
        ax = parallel(self.df, legend=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
