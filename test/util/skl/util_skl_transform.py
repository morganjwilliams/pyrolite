import unittest
from pyrolite.util.synthetic import normal_frame
from pyrolite.comp.codata import close
from pyrolite.geochem.ind import REE

try:
    import sklearn
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    HAVE_SKLEARN = True

    def test_classifier():
        param_grid = dict(gamma=np.array([0.001, 0.01]), C=np.array([1, 10]))
        gs = GridSearchCV(SVC(gamma="scale"), param_grid, cv=2)
        return gs

    from pyrolite.util.skl.transform import *

except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestLogTransformers(unittest.TestCase):
    """Checks the scikit-learn invertible transformer classes."""

    def setUp(self):
        self.df = normal_frame().apply(close, axis=1)

    def test_linear_transformer(self):
        """Test the linear transfomer."""
        df = self.df
        tmr = LinearTransform()
        for input in [df, df.values]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_exp_transformer(self):
        """Test the linear transfomer."""
        df = self.df
        tmr = ExpTransform()
        for input in [df, df.values, df.iloc[0, :]]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_exp_transformer(self):
        """Test the linear transfomer."""
        df = self.df
        tmr = LogTransform()
        for input in [df, df.values, df.iloc[0, :]]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_ALR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = ALRTransform()
        for input in [df, df.values, df.iloc[0, :]]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_CLR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = CLRTransform()
        for input in [df, df.values, df.iloc[0, :]]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_ILR_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = ILRTransform()
        for input in [df, df.values, df.iloc[0, :]]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_BoxCox_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = BoxCoxTransform()
        for input in [df, df.values, df.iloc[0, :]]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestAgumentors(unittest.TestCase):
    """Checks the default config for scikit-learn augmenting transformer classes."""

    def setUp(self):
        self.df = normal_frame().apply(close, axis=1)

    def test_DropBelowZero(self):
        """Test the DropBelowZero transfomer."""
        df = self.df
        tmr = DropBelowZero()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_Devolatilizer(self):
        """Test the Devolatilizer transfomer."""
        df = self.df
        tmr = Devolatilizer()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_ElementAggregator(self):
        """Test the ElementAggregator transfomer."""
        df = self.df
        tmr = ElementAggregator()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_LambdaTransformer(self):
        """Test the LambdaTransformer transfomer."""
        df = normal_frame(columns=REE()).apply(close, axis=1)
        tmr = LambdaTransformer()
        for ree in [REE(), [i for i in REE() if i not in ["Eu"]]]:
            with self.subTest(ree=ree):
                out = tmr.transform(df.loc[:, ree])


if __name__ == "__main__":
    unittest.main()
