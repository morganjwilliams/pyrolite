import unittest
import pandas as pd
import numpy as np
from pyrolite.util.spatial import *


class TestPieceWise(unittest.TestCase):
    def test_pieces(self):
        x1, x2 = 0.0, 10.0
        segment_ranges = [(x1, x2)]
        for segments in [1, 2, 3]:
            with self.subTest(segments=segments):
                result = list(piecewise(segment_ranges, segments=segments))
                self.assertTrue(len(result) == segments)

    def test_multiple_ranges(self):
        x1, x2 = 0.0, 10.0
        segment_ranges = [(x1, x2), (x2, x1), (x1, x2)]
        segments = 2
        result = list(piecewise(segment_ranges, segments=segments))
        self.assertTrue(len(result) == segments ** len(segment_ranges))


class TestSpatioTemporalSplit(unittest.TestCase):
    def test_split(self):
        x1, x2 = 0, 10
        segments = 2
        params = dict(age=(0, 10), lat=(-10, 10), lo=(-90, 90))
        result = list(spatiotemporal_split(segments=segments, **params))

        self.assertTrue([isinstance(item, dict) for item in result])
        self.assertTrue(len(result) == segments ** len(params))


class TestNSEW2Bounds(unittest.TestCase):
    def setUp(self):
        self.params = {
            k: v
            for (k, v) in zip(
                ["west", "south", "east", "north"], np.random.randint(1, 10, 4)
            )
        }

    def test_conversion(self):
        result = NSEW_2_bounds(self.params)
        self.assertTrue(isinstance(result, list))

    def test_order(self):
        order = ["minx", "maxx", "miny", "maxy"]
        result = NSEW_2_bounds(self.params, order=order)
        self.assertTrue(result[1] == self.params["east"])


if __name__ == "__main__":
    unittest.main()
