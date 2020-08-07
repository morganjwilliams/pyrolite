import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs

    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False
from pyrolite.util.spatial import *
from pyrolite.util.math import isclose  # nan-equalling isclose


class TestGreatCircleDistance(unittest.TestCase):
    def setUp(self):

        self.ps = zip(
            np.array(
                [
                    ([0, 0], [0, 0]),  # should be 0
                    ([-170, 0], [170, 0]),  # should be 20
                    ([0, -90], [0, 90]),  # should be 180
                    ([-45, 0], [45.0, 0.0]),  # should be 90
                    ([-90, -90], [90.0, 90.0]),  # should be 180
                    ([-90, -45], [90.0, 45.0]),  # should be 180, rotation of above
                    ([-90, -0], [90.0, 0.0]),  # should be 180, rotation of above
                    ([-60, 20], [45.0, 15.0]),
                    ([-87.0, 67.0], [34, 14]),
                    ([-45, -45], [45.0, 45.0]),
                    ([-45, -30], [45.0, 30.0]),
                ]
            ),
            [0, 20, 180, 90, 180, 180, 180, None, None, None, None],
        )

    def test_default(self):
        for ps, expect in self.ps:
            with self.subTest(ps=ps, expect=expect):
                distance = great_circle_distance(*ps)
                distance_r = great_circle_distance(*ps[::-1])
                self.assertTrue(isclose(distance, distance_r))
                if (ps[0] == ps[1]).all():
                    self.assertTrue(np.isclose(distance, 0.0))

        """
        ax = plt.subplot(111, projection=ccrs.Mollweide())  # ccrs.Orthographic(0, 0))
        ax.figure.set_size_inches(8, 8)
        ax.stock_img()

        ax.plot(
            *np.array([*ps]).T,
            color="blue",
            marker="o",
            transform=ccrs.Geodetic()
        )
        ax.plot(*np.array([*ps]).T, color="gray", transform=ccrs.PlateCarree())
        plt.text(
            **np.array([*ps])[0] + [5, 5],
            "{:2.0f}".format(distance),
            horizontalalignment="left",
            fontsize=10,
            transform=ccrs.Geodetic()
        )
        plt.show()"""

    def test_absolute(self):
        for ps, expect in self.ps:
            for absolute in [True, False]:
                with self.subTest(ps=ps, expect=expect, absolute=absolute):
                    distance = great_circle_distance(*ps, absolute=absolute)
                    distance_r = great_circle_distance(*ps[::-1], absolute=absolute)
                    self.assertTrue(isclose(distance, distance_r))
                    if (ps[0] == ps[1]).all():
                        self.assertTrue(np.isclose(distance, 0.0))

    def test_degrees(self):
        for ps, expect in self.ps:
            for degrees in [True, False]:
                with self.subTest(ps=ps, expect=expect, degrees=degrees):
                    if not degrees:
                        ps = np.deg2rad(
                            ps
                        )  # convert to radians to give sensible output
                    distance = great_circle_distance(*ps, degrees=degrees)
                    distance_r = great_circle_distance(*ps[::-1], degrees=degrees)
                    self.assertTrue(isclose(distance, distance_r))
                    if (ps[0] == ps[1]).all():
                        self.assertTrue(np.isclose(distance, 0.0))
                    if expect is not None:
                        self.assertTrue(isclose(distance, expect))

    def test_Vicenty(self):
        method = "vicenty"
        for ps, expect in self.ps:
            with self.subTest(ps=ps, expect=expect, method=method):
                distance = great_circle_distance(*ps, method=method)
                distance_r = great_circle_distance(*ps[::-1], method=method)
                self.assertTrue(isclose(distance, distance_r))
                if (ps[0] == ps[1]).all():
                    self.assertTrue(np.isclose(distance, 0.0))
                if expect is not None:
                    self.assertTrue(isclose(distance, expect))

    def test_haversine(self):
        method = "haversine"
        for ps, expect in self.ps:
            with self.subTest(ps=ps, expect=expect, method=method):
                distance = great_circle_distance(*ps, method=method)
                distance_r = great_circle_distance(*ps[::-1], method=method)
                self.assertTrue(isclose(distance, distance_r))
                if (ps[0] == ps[1]).all():
                    self.assertTrue(np.isclose(distance, 0.0))
                if expect is not None:
                    self.assertTrue(isclose(distance, expect))

    def test_cosines(self):
        method = "cosines"
        for ps, expect in self.ps:
            with self.subTest(ps=ps, expect=expect, method=method):
                distance = great_circle_distance(*ps, method=method)
                distance_r = great_circle_distance(*ps[::-1], method=method)
                self.assertTrue(isclose(distance, distance_r))
                if (ps[0] == ps[1]).all():
                    self.assertTrue(np.isclose(distance, 0.0))
                if expect is not None:
                    self.assertTrue(isclose(distance, expect))


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


class TestLevenshteinDistance(unittest.TestCase):
    def test_string(self):
        pairs = [
            ("bar", "car"),
            ("bart", "car"),
            ("Saturday", "Sunday"),
            ("kitten", "sitting"),
        ]
        expect = [1, 2, 3, 3]
        for pair, exp in zip(pairs, expect):
            with self.subTest(pair=pair, exp=exp):
                dist = levenshtein_distance(*pair)
                self.assertTrue(dist == exp)

    def test_list(self):
        pairs = [
            ([1, 2, 3], [1, 2, 2]),
            (["A", "B", "C"], ["A", "B"]),
            (["A", "B", "C", "D"], ["A", "E", "C"]),
        ]
        expect = [1, 1, 2]
        for pair, exp in zip(pairs, expect):
            with self.subTest(pair=pair, exp=exp):
                dist = levenshtein_distance(*pair)
                self.assertTrue(dist == exp)


if __name__ == "__main__":
    unittest.main()
