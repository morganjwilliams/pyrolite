import io
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pyrolite.util.general import remove_tempdir
from pyrolite.util.plot.export import (
    get_full_extent,
    path_to_csv,
    save_axes,
    save_figure,
)


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
        for save_at in [self.tempdir, str(self.tempdir), self.tempdir / "subfolder"]:
            for save_fmts in [("png",), ("png", "svg")]:
                with self.subTest(save_at=save_at, save_fmts=save_fmts):
                    save_figure(
                        self.fig, name="test_fig", save_at=save_at, save_fmts=save_fmts
                    )

    def test_save_axes(self):
        self.ax.set(title="A title", xlabel="xaxis", ylabel="yaxis")
        for pad in [0, 0.1, (0.1, 0.1)]:
            for ax in [self.ax, plt.subplots(3)[1][:2]]:  # test multi-axes too
                with self.subTest(pad=pad):
                    save_axes(ax, name="test_ax", save_at=str(self.tempdir), pad=pad)

    def tearDown(self):
        remove_tempdir(str(self.tempdir))
        plt.close("all")


class TestPath2CSV(unittest.TestCase):

    def setUp(self):
        self.fig, self.ax = plt.subplots(1)
        self.ax.plot([0, 1, 2, 3], [0, 0, 0, 0])
        self.path = self.ax.lines[0].get_path()

    def test_path_to_csv(self):
        csvdata = path_to_csv(self.path)
        df = pd.read_csv(io.StringIO(csvdata))
        print(df.columns)
        self.assertTrue((df["y"] == 0).all())

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
