import unittest

from pyrolite.classification import *

from pyrolite.compositions import renormalise

class TestTAS(unittest.TestCase):


    def test_classifer_build(self):
        cm = Geochemistry.TAS(rebuild=False)

    def test_classifer_plot(self):
        cm = Geochemistry.TAS(rebuild=False)
        fig, ax = plt.subplots(1)
        cm.add_to_axes(ax=ax, alpha=0.4, color='k')

    def test_alkalinity_classifier(self):
        c2 = Geochemistry.peralkalinity()
        df = pd.DataFrame(index=np.arange(100))
        df['Al2O3'] = np.linspace(20, 50, len(df.index))
        df['Na2O'] = np.linspace(20, 10, len(df.index))
        df['K2O'] = np.linspace(10, 122, len(df.index))
        df['CaO'] = np.linspace(5, 30, len(df.index))
        df = renormalise(df)
        c2.classify(df)


if __name__ == '__main__':
    unittest.main()
