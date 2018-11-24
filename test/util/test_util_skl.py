import unittest
import numpy as np
from pyrolite.util.pd import test_df
from pyrolite.comp.renorm import close
from pyrolite.util.skl import *


class TestLogTransformers(unittest.TestCase):
    """Checks the scikit-learn transformer classes."""

    def setUp(self):
        self.df = test_df().apply(close, axis=1)

    def test_linear_transformer(self):
        """Test the linear transfomer."""
        df = self.df
        tmr = LinearTransform()
        for input in [df, df.values]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_ALR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = ALRTransform()
        for input in [df, df.values]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_CLR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = CLRTransform()
        for input in [df, df.values]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_ILR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = ILRTransform()
        for input in [df, df.values]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_BoxCoX_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = BoxCoxTransform()
        for input in [df, df.values]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))


if __name__ == "__main__":
    unittest.main()
