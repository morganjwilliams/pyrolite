import unittest
import matplotlib.pyplot as plt
from pyrolite.plot.templates.AIOCG import AIOCG


class TestAIOCGPlot(unittest.TestCase):
    def setUp(self):
        pass

    def test_TAS_default(self):
        fig, axes = plt.subplots(1)
        for ax in [None, axes]:
            with self.subTest(ax=ax):
                ax = AIOCG(ax)

    def test_TAS_relim(self):
        for relim in [True, False]:
            with self.subTest(relim=relim):
                ax = AIOCG(relim=relim)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
