import unittest
from pathlib import Path
import matplotlib.pyplot as plt
from pyrolite.util.general import remove_tempdir
from pyrolite.util.plot.export import get_full_extent, save_axes, save_figure


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
