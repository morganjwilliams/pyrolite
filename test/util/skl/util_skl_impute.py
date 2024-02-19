import unittest

import numpy as np

from pyrolite.comp.codata import close
from pyrolite.util.synthetic import normal_frame

try:
    import sklearn

    HAVE_SKLEARN = True

    from pyrolite.util.skl.impute import MultipleImputer
except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestImputers(unittest.TestCase):
    """Checks the default config for scikit-learn imputing transformer classes."""

    def setUp(self):
        self.df = normal_frame().apply(close, axis=1)
        self.group = (self.df["MgO"] > 0.21).apply(int)

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
