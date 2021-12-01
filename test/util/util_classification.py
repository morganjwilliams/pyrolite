import unittest

import matplotlib.pyplot as plt
import numpy as np

from pyrolite.comp.codata import renormalise
from pyrolite.util.classification import *
from pyrolite.util.synthetic import normal_frame


class TestTAS(unittest.TestCase):
    """Test the TAS classifier."""

    def setUp(self):
        self.df = normal_frame(
            columns=["SiO2", "Na2O", "K2O", "Al2O3"],
            mean=[0.5, 0.04, 0.05, 0.4],
            size=100,
        )
        self.df.loc[:, "Na2O + K2O"] = self.df.Na2O + self.df.K2O

    def test_classifer_build(self):
        cm = TAS()

    def test_classifer_add_to_axes(self):
        cm = TAS()
        fig, ax = plt.subplots(1)
        for a in [None, ax]:
            with self.subTest(a=a):
                cm.add_to_axes(
                    ax=a,
                    alpha=0.4,
                    color="k",
                    axes_scale=100,
                    linewidth=0.5,
                    which_labels="ID",
                    add_labels=True,
                )

    def test_classifer_predict(self):
        df = self.df
        cm = TAS()
        classes = cm.predict(df, data_scale=1.0)
        # precitions will be ID's
        rocknames = classes.apply(lambda x: cm.fields.get(x, {"name": None})["name"])
        self.assertFalse(pd.isnull(classes).all())


class TestUSDASoilTexture(unittest.TestCase):
    def setUp(self):
        self.df = normal_frame(
            columns=["Sand", "Clay", "Silt"],
            mean=[0.5, 0.05, 0.45],
            size=100,
        )

    def test_classifer_build(self):
        cm = USDASoilTexture()

    def test_classifer_add_to_axes(self):
        cm = USDASoilTexture()
        fig, ax = plt.subplots(1)
        for a in [None, ax]:
            with self.subTest(a=a):
                cm.add_to_axes(
                    ax=a, alpha=0.4, color="k", linewidth=0.5, add_labels=True
                )

    def test_classifer_predict(self):
        df = self.df
        cm = USDASoilTexture()
        classes = cm.predict(df, data_scale=1.0)
        self.assertFalse(pd.isnull(classes).all())


class TestQAP(unittest.TestCase):
    def setUp(self):
        self.df = normal_frame(
            columns=["Quartz", "Alkali Feldspar", "Plagioclase"],
            mean=[0.5, 0.05, 0.45],
            size=100,
        )

    def test_classifer_build(self):
        cm = QAP()

    def test_classifer_add_to_axes(self):
        cm = QAP()
        fig, ax = plt.subplots(1)
        for a in [None, ax]:
            with self.subTest(a=a):
                cm.add_to_axes(
                    ax=a, alpha=0.4, color="k", linewidth=0.5, add_labels=True
                )

    def test_classifer_predict(self):
        df = self.df
        cm = QAP()
        classes = cm.predict(df)
        self.assertFalse(pd.isnull(classes).all())


class TestFeldsparTernary(unittest.TestCase):
    def setUp(self):
        self.df = normal_frame(
            columns=["Ca", "Na", "K"],
            mean=[0.5, 0.05, 0.45],
            size=100,
        )

    def test_classifer_build(self):
        cm = FeldsparTernary()

    def test_classifer_add_to_axes(self):
        cm = FeldsparTernary()
        fig, ax = plt.subplots(1)
        for a in [None, ax]:
            with self.subTest(a=a):
                cm.add_to_axes(
                    ax=a, alpha=0.4, color="k", linewidth=0.5, add_labels=True
                )

    def test_classifer_predict(self):
        df = self.df
        cm = FeldsparTernary()
        classes = cm.predict(df)
        self.assertFalse(pd.isnull(classes).all())


class TestPeralkalinity(unittest.TestCase):
    """Test the peralkalinity classifier."""

    def setUp(self):
        self.df = df = normal_frame(
            columns=["SiO2", "Na2O", "K2O", "Al2O3", "CaO"],
            mean=[0.5, 0.04, 0.05, 0.2, 0.3],
            size=100,
        )

    def test_classifer_predict(self):
        df = self.df
        cm = PeralkalinityClassifier()
        df.loc[:, "Peralk"] = cm.predict(df)


if __name__ == "__main__":
    unittest.main()
