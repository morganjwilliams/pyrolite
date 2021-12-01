import unittest

import matplotlib.pyplot as plt

from pyrolite.plot.templates import (QAP, TAS, FeldsparTernary,
                                     USDASoilTexture, pearceThNbYb,
                                     pearceTiNbYb)


class TestTASPlot(unittest.TestCase):
    def setUp(self):
        pass

    def test_TAS_default(self):
        fig, axes = plt.subplots(1)
        for ax in [None, axes]:
            with self.subTest(ax=ax):
                ax = TAS(ax)

    def test_TAS_relim(self):
        for relim in [True, False]:
            with self.subTest(relim=relim):
                ax = TAS(relim=relim)

    def tearDown(self):
        plt.close("all")


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


class TestTernaryDiagrams(unittest.TestCase):
    def test_ternary_default(self):
        for diagram in [QAP, FeldsparTernary, USDASoilTexture]:
            fig, axes = plt.subplots(1)
            for a in [None, axes]:
                with self.subTest(diagram=diagram, a=a):
                    ax = diagram(a)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
