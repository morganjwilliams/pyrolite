import unittest

import matplotlib.pyplot as plt
from pyrolite.plot.templates import (
    QAP,
    TAS,
    FeldsparTernary,
    Herron,
    JensenPlot,
    Pettijohn,
    SpinelFeBivariate,
    SpinelTrivalentTernary,
    USDASoilTexture,
    pearceThNbYb,
    pearceTiNbYb,
)


class TestTASPlot(unittest.TestCase):
    def setUp(self):
        pass

    def test_TAS_default(self):
        fig, axes = plt.subplots(1)
        for ax in [None, axes]:
            for variant in [None, "Middlemost", "LeMaitre", "LeMaitreCombined"]:
                with self.subTest(ax=ax, variant=variant):
                    ax = TAS(ax, which_model=variant)

    def test_TAS_variants(self):
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


class TestDiagramsGeneral(unittest.TestCase):
    def test_bivariate_diagrams(self):
        for diagram in [Pettijohn, Herron, SpinelFeBivariate]:
            fig, axes = plt.subplots(1)
            for a in [None, axes]:
                with self.subTest(diagram=diagram, a=a):
                    ax = diagram(a)
            plt.close(fig)

    def test_ternary_diagrams(self):
        for diagram in [
            QAP,
            FeldsparTernary,
            USDASoilTexture,
            JensenPlot,
            SpinelTrivalentTernary,
        ]:
            fig, axes = plt.subplots(1)
            for a in [None, axes]:
                with self.subTest(diagram=diagram, a=a):
                    ax = diagram(a)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
