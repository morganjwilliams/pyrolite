import unittest
import numpy as np
from pyrolite.comp.codata import *
from pyrolite.util.pd import test_df


class TestALR(unittest.TestCase):
    """Test the numpy additive log ratio transformation."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.cols, np.random.rand(len(self.cols), 10))}
        )
        self.df = self.df.apply(lambda x: x / np.sum(x), axis="columns")

    def test_single(self):
        """Checks whether the function works on a single record."""
        df = self.df.head(1)
        out = alr(df.values)

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        df = self.df
        out = alr(df.values)

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        df = self.df.head(1)
        out = alr(df.values)
        inv = inv_alr(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        df = self.df
        out = alr(df.values)
        inv = inv_alr(out)
        self.assertTrue(np.allclose(inv, df.values))


class TestCLR(unittest.TestCase):
    """Test the centred log ratio transformation."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.cols, np.random.rand(len(self.cols), 10))}
        )
        self.df = self.df.apply(lambda x: x / np.sum(x), axis="columns")

    def test_single(self):
        """Checks whether the function works on a single record."""
        df = self.df.head(1)
        out = clr(df.values)

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        df = self.df
        out = clr(df.values)

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        df = self.df.head(1)
        out = clr(df.values)
        inv = inv_clr(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        df = self.df
        out = clr(df.values)
        inv = inv_clr(out)
        self.assertTrue(np.allclose(inv, df.values))


class TestILR(unittest.TestCase):
    """Test the isometric log ratio transformation."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.cols, np.random.rand(len(self.cols), 10))}
        )
        self.df = self.df.apply(lambda x: x / np.sum(x), axis="columns")

    def test_single(self):
        """Checks whether the function works on a single record."""
        df = self.df.head(1)
        out = ilr(df.values)

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        df = self.df
        out = ilr(df.values)

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        df = self.df.head(1)
        out = ilr(df.values)
        inv = inv_ilr(out, X=df.values)
        self.assertTrue(np.allclose(inv, df.values))

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        df = self.df
        out = ilr(df.values)
        inv = inv_ilr(out, X=df.values)
        self.assertTrue(np.allclose(inv, df.values))


class TestLogTransformers(unittest.TestCase):
    """Checks the scikit-learn transformer classes."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2"]
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.cols, np.random.rand(len(self.cols), 10))}
        )
        self.df = self.df.apply(lambda x: x / np.sum(x), axis="columns")

    def test_linear_transformer(self):
        """Test the linear transfomer."""
        df = self.df
        tmr = LinearTransform()
        out = tmr.transform(df.values)
        inv = tmr.inverse_transform(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_ALR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = ALRTransform()
        out = tmr.transform(df.values)
        inv = tmr.inverse_transform(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_CLR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = CLRTransform()
        out = tmr.transform(df.values)
        inv = tmr.inverse_transform(out)
        self.assertTrue(np.allclose(inv, df.values))

    def test_ILR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = ILRTransform()
        out = tmr.transform(df.values)
        inv = tmr.inverse_transform(out)
        self.assertTrue(np.allclose(inv, df.values))


if __name__ == "__main__":
    unittest.main()
