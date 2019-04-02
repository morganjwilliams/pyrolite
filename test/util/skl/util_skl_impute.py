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
        gs = GridSearchCV(SVC(gamma='scale'), param_grid, cv=2)
        return gs


except ImportError:
    HAVE_SKLEARN = False

if HAVE_SKLEARN:
    from pyrolite.util.skl import *

try:
    from fancyimpute import SoftImpute, IterativeImputer

    HAVE_IMPUTE = True
except ImportError:
    HAVE_IMPUTE = False


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
@unittest.skipUnless(HAVE_IMPUTE, "Requires fancyimpute")
class TestImputers(unittest.TestCase):
    """Checks the default config for scikit-learn imputing transformer classes."""

    def setUp(self):
        self.df = test_df().apply(close, axis=1)

    def test_MultipleImputer(self):
        """Test the MultipleImputer transfomer."""
        df = self.df
        tmr = MultipleImputer()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.fit_transform(input)

    def test_PdSoftImputer(self):
        """Test the PdSoftImputer transfomer."""
        df = self.df
        tmr = PdSoftImputer()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.fit_transform(input)


if __name__ == "__main__":
    unittest.main()
