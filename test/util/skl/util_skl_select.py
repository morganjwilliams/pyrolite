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

    from pyrolite.util.skl.select import *

except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestSelectors(unittest.TestCase):
    """Checks the default config for scikit-learn selector classes."""

    def setUp(self):
        self.df = normal_frame().apply(close, axis=1)

    def test_ColumnSelector(self):
        """Test the ColumnSelector transfomer."""
        df = self.df
        tmr = ColumnSelector(columns=self.df.columns)
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_TypeSelector(self):
        """Test the TypeSelector transfomer."""
        df = self.df
        tmr = TypeSelector(np.float)
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_CompositionalSelector(self):
        """Test the CompositionalSelector transfomer."""
        df = self.df
        tmr = CompositionalSelector()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_MajorsSelector(self):
        """Test the MajorsSelector transfomer."""
        df = self.df
        tmr = MajorsSelector()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_ElementSelector(self):
        """Test the ElementSelector transfomer."""
        df = self.df
        tmr = ElementSelector()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_REESelector(self):
        """Test the REESelector transfomer."""
        df = self.df
        tmr = REESelector()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)


if __name__ == "__main__":
    unittest.main()
