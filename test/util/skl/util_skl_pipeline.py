import unittest
from pyrolite.util.synthetic import test_df
from pyrolite.comp.codata import close
try:
    import sklearn
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    HAVE_SKLEARN = True

    def test_classifier():
        param_grid = dict(gamma=np.array([0.001, 0.01]), C=np.array([1, 10]))
        gs = GridSearchCV(SVC(), param_grid, cv=2)
        return gs

    from pyrolite.util.skl.pipeline import *

except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestPdUnion(unittest.TestCase):
    """Checks the default config for scikit-learn augmenting transformer classes."""

    def setUp(self):
        self.df = test_df().apply(close, axis=1)

    def test_PdUnion(self):
        """Test the PdUnion pipeline augmentor."""
        df = self.df
        tmr = PdUnion()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

if __name__ == '__main__':
    unittest.main()
