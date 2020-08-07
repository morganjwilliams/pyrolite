import unittest
import numpy as np
from pyrolite.util.synthetic import normal_frame
from pyrolite.comp.codata import close

try:
    import sklearn
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    HAVE_SKLEARN = True

    def test_classifier():
        param_grid = dict(gamma=np.array([0.001, 0.01]), C=np.array([1, 10]))
        gs = GridSearchCV(SVC(gamma="scale"), param_grid, cv=2)
        return gs


except ImportError:
    HAVE_SKLEARN = False

if HAVE_SKLEARN:
    from pyrolite.util.skl.impute import MultipleImputer


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestImputers(unittest.TestCase):
    """Checks the default config for scikit-learn imputing transformer classes."""

    def setUp(self):
        self.df = normal_frame().apply(close, axis=1)
        self.group = (self.df["MgO"] > 0.21).apply(np.int)

    def test_MultipleImputer(self):
        """Test the MultipleImputer transfomer."""
        df = self.df
        tmr = MultipleImputer()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.fit_transform(input)

    def test_groupby(self):
        df = self.df
        df["group"] = self.group
        tmr = MultipleImputer(groupby="group")
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.fit_transform(input)

    def test_y_specified(self):
        df = self.df
        tmr = MultipleImputer()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.fit_transform(input, y=self.group)

    @unittest.expectedFailure
    def test_groupby_with_y_specified(self):
        df = self.df
        df["group"] = self.group
        tmr = MultipleImputer(groupby="group")
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.fit_transform(input, y=self.group)


if __name__ == "__main__":
    unittest.main()
