import unittest
from pyrolite.util.classification import *
from pyrolite.comp.codata import renormalise


class TestTAS(unittest.TestCase):
    """Test the TAS classifier."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2", "Na2O", "K2O", "Al2O3"]
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.cols, np.random.rand(len(self.cols), 10))}
        )
        self.df.loc[:, "TotalAlkali"] = self.df.Na2O + self.df.K2O

    def test_classifer_norebuild(self):
        cm = Geochemistry.TAS(rebuild=False)

    def test_classifer_rebuild(self):
        cm = Geochemistry.TAS(rebuild=True)

    def test_classifer_plot(self):
        cm = Geochemistry.TAS(rebuild=True)
        fig, ax = plt.subplots(1)
        cm.add_to_axes(ax=ax, alpha=0.4, color="k")

    def test_classifer_classify(self):
        df = self.df
        df = renormalise(df)
        cm = Geochemistry.TAS(rebuild=True)
        df.loc[:, "TAS"] = cm.classify(df)
        df["Rocknames"] = df.TAS.apply(
            lambda x: cm.clsf.fields.get(x, {"names": None})["names"]
        )


class TestPeralkalinity(unittest.TestCase):
    """Test the peralkalinity classifier."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2", "Na2O", "K2O", "Al2O3"]
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.cols, np.random.rand(len(self.cols), 10))}
        )

    def test_classifer_rebuild(self):
        cm = Geochemistry.peralkalinity(rebuild=True)

    def test_classifer_classify(self):
        df = self.df
        df = renormalise(df)
        cm = Geochemistry.peralkalinity(rebuild=True)
        df.loc[:, "Peralk"] = cm.classify(df)


if __name__ == "__main__":
    unittest.main()
