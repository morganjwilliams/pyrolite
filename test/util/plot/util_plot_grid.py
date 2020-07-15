import unittest
import numpy as np
from pyrolite.util.plot.grid import (
    bin_centres_to_edges,
    bin_edges_to_centres,
    ternary_grid,
)
from pyrolite.util.synthetic import normal_frame


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


class TestTernaryGrid(unittest.TestCase):

    def setUp(self):
        self.data = normal_frame(columns=["SiO2", "CaO", "MgO"], size=20).values

    def test_default(self):
        # no data supplied, grid will cover ternary space up until some margin
        bins, binedges, centregrid, edgegrid = ternary_grid()

    def test_data_unforced_margin(self):
        # where you supply data and you want to build a grid covering it.
        bins, binedges, centregrid, edgegrid = ternary_grid(data=self.data)

    def test_data_forced_margin(self):
        # where you supply data and you want to build a grid covering it,
        # but enforce the margin (such that some data might fall outside the grid)
        bins, binedges, centregrid, edgegrid = ternary_grid(
            data=self.data, force_margin=True
        )


if __name__ == "__main__":
    unittest.main()
