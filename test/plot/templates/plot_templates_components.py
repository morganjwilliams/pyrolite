import unittest
import numpy as np
import matplotlib.pyplot as plt
from pyrolite.plot.templates.components import *


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.xy = [1.0, 2.0]

    def test_default(self):
        pt = Point(self.xy)

    def test_add_to_axes(self):
        fig, ax = plt.subplots(1)
        pt = Point(self.xy)
        pt.add_to_axes(ax)

    def tearDown(self):
        plt.close("all")


class TestLinear2D(unittest.TestCase):
    def setUp(self):
        self.xy = [1.0, 2.0]
        self.xy2 = [2.0, 3.0]
        self.slope = 1.0

    def test_slope_construct(self):
        line = Linear2D(self.xy, slope=self.slope)

    def test_p1_construct(self):
        line = Linear2D(self.xy, p1=self.xy2)

    def test_add_to_axes(self):
        fig, ax = plt.subplots(1)
        line = Linear2D(self.xy, slope=self.slope)
        line.add_to_axes(ax)

    def test_invert_axes(self):
        line = Linear2D(self.xy, slope=self.slope)
        line.invert_axes()

    def test_perpendicular_line(self):
        line = Linear2D(self.xy, slope=self.slope)
        perpline = line.perpendicular_line(self.xy)

    def test_perpendicular_intersect(self):
        line = Linear2D(self.xy, slope=self.slope)
        perpline = line.perpendicular_line(self.xy)
        line.intersect(perpline)

    def tearDown(self):
        plt.close("all")


class TestLogLinear2D(unittest.TestCase):
    def setUp(self):
        self.xy = [1.0, 2.0]
        self.xy2 = [2.0, 3.0]
        self.slope = 1.0

    def test_slope_construct(self):
        line = LogLinear2D(self.xy, slope=self.slope)

    def test_p1_construct(self):
        line = LogLinear2D(self.xy, p1=self.xy2)

    def test_add_to_axes(self):
        fig, ax = plt.subplots(1)
        line = LogLinear2D(self.xy, p1=self.xy2)
        line.add_to_axes(ax)

    def test_invert_axes(self):
        line = LogLinear2D(self.xy, p1=self.xy2)
        line.invert_axes()

    def test_perpendicular_line(self):
        line = LogLinear2D(self.xy, slope=self.slope)
        perpline = line.perpendicular_line(self.xy)

    def test_perpendicular_intersect(self):
        line = LogLinear2D(self.xy, slope=self.slope)
        perpline = line.perpendicular_line(self.xy)
        line.intersect(perpline)

    def tearDown(self):
        plt.close("all")


class TestGeometryCollection(unittest.TestCase):
    def setUp(self):
        self.points = [
            Point([x, y]) for x, y in zip(np.random.randn(10), np.random.randn(10))
        ]

        self.lines = [
            Linear2D([x, y], slope=1.0)
            for x, y in zip(np.random.randn(10), np.random.randn(10))
        ]

    def test_default(self):
        geom = GeometryCollection(*(self.points + self.lines))

    def test_add_to_axes(self):
        fig, ax = plt.subplots(1)
        geom = GeometryCollection(*(self.points + self.lines))
        geom.add_to_axes(ax)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
