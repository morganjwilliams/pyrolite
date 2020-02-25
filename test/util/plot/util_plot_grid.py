import unittest
import numpy as np
from pyrolite.util.plot.grid import (
    bin_centres_to_edges,
    bin_edges_to_centres,
    ternary_grid,
)


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


if __name__ == "__main__":
    unittest.main()
