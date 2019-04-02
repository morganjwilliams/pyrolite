import unittest
import matplotlib.pyplot as plt
from pyrolite.plot.templates.pearce import pearceThNbYb, pearceTiNbYb


class TestPearcePlots(unittest.TestCase):
    def setUp(self):
        pass

    def test_pearceThNbYb_default(self):
        fig, axes = plt.subplots(1)
        for ax in [None, axes]:
            with self.subTest(ax=ax):
                ax = pearceThNbYb(ax)

    def test_pearceThNbYb_relim(self):
        for relim in [True, False]:
            with self.subTest(relim=relim):
                ax = pearceThNbYb(relim=relim)

    def test_pearceTibYb_default(self):
        fig, axes = plt.subplots(1)
        for ax in [None, axes]:
            with self.subTest(ax=ax):
                ax = pearceTiNbYb(ax)

    def test_pearceThNbYb_relim(self):
        for relim in [True, False]:
            with self.subTest(relim=relim):
                ax = pearceThNbYb(relim=relim)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
