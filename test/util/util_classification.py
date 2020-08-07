import unittest
from pyrolite.util.classification import *
from pyrolite.comp.codata import renormalise
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
        cm.add_to_axes(
            ax=ax, alpha=0.4, color="k", axes_scale=100, linewidth=0.5, labels="ID",
        )

    def test_classifer_new_axes(self):
        cm = TAS()
        cm.add_to_axes(
            alpha=0.4,
            color="k",
            axes_scale=100,
            linewidth=0.5,
            labels="ID",
            figsize=(10, 8),
        )

    def test_classifer_predict(self):
        df = self.df
        cm = TAS()
        df.loc[:, "TAS"] = cm.predict(df, data_scale=1.0)
        # precitions will be ID's
        df["Rocknames"] = df.TAS.apply(
            lambda x: cm.fields.get(x, {"name": None})["name"]
        )
        self.assertFalse(pd.isnull(df["TAS"]).all())


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
