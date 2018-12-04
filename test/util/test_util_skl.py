import unittest
import numpy as np
from pyrolite.util.pd import test_df
from pyrolite.comp.renorm import close
from pyrolite.geochem import REE

try:
    import sklearn
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, train_test_split

    from pyrolite.util.skl import (
        LinearTransform,
        ExpTransform,
        LogTransform,
        ALRTransform,
        CLRTransform,
        ILRTransform,
        BoxCoxTransform,
        BoxCoxTransform,
        DropBelowZero,
        ColumnSelector,
        TypeSelector,
        CompositionalSelector,
        MajorsSelector,
        ElementSelector,
        REESelector,
        Devolatilizer,
        RedoxAggregator,
        ElementAggregator,
        PdUnion,
        LambdaTransformer,
        MultipleImputer,
        PdSoftImputer,
        get_confusion_matrix,
        plot_confusion_matrix,
        plot_gs_results,
    )

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

try:
    import imblearn

    HAVE_IMBLEARN = True
except ImportError:
    HAVE_IMBLEARN = False

try:
    from fancyimpute import SoftImpute, IterativeImputer

    HAVE_IMPUTE = True
except ImportError:
    HAVE_IMPUTE = False


def test_classifier():
    param_grid = dict(gamma=np.array([0.001, 0.01]), C=np.array([1, 10]))
    gs = GridSearchCV(SVC(), param_grid, cv=2)
    return gs


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestGetConfusionMatrix(unittest.TestCase):
    """Checks the confusion matrix function."""

    def setUp(self):
        self.X = test_df(index_length=20).apply(close, axis=1)
        self.gs = test_classifier()
        self.y = np.ones(self.X.index.size)
        self.y[4:] += 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y
        )
        self.gs.fit(self.X_train, self.y_train)
        self.clf = self.gs.best_estimator_

    def test_confusion_matrix(self):
        get_confusion_matrix(self.clf, self.X_test, self.y_test)


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestPlotConfusionMatrix(unittest.TestCase):
    """Checks the plot_confusion_matrix matrix function."""

    def setUp(self):
        self.X = test_df(index_length=20).apply(close, axis=1)
        self.gs = test_classifier()
        self.y = np.ones(self.X.index.size)
        self.y[4:] += 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y
        )
        self.gs.fit(self.X_train, self.y_train)
        self.clf = self.gs.best_estimator_

    def test_plot(self):
        plot_confusion_matrix(self.gs, self.X_test, self.y_test)


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestPlotGSResults(unittest.TestCase):
    """Checks the plot_gs_results matrix function."""

    def setUp(self):
        self.X = test_df(index_length=20).apply(close, axis=1)
        self.gs = test_classifier()
        self.y = np.ones(self.X.index.size)
        self.y[4:] += 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y
        )
        self.gs.fit(self.X_train, self.y_train)
        self.clf = self.gs.best_estimator_

    def test_plot(self):
        plot_gs_results(self.gs)


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestLogTransformers(unittest.TestCase):
    """Checks the scikit-learn invertible transformer classes."""

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

    def test_exp_transformer(self):
        """Test the linear transfomer."""
        df = self.df
        tmr = ExpTransform()
        for input in [df, df.values]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))

    def test_exp_transformer(self):
        """Test the linear transfomer."""
        df = self.df
        tmr = LogTransform()
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

    def test_BoxCox_transformer(self):
        """Test the isometric log ratio transfomer."""
        df = self.df
        tmr = BoxCoxTransform()
        for input in [df, df.values]:
            with self.subTest(input=input):
                out = tmr.transform(input)
                inv = tmr.inverse_transform(out)
                self.assertTrue(np.allclose(np.array(inv), np.array(input)))


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestSelectors(unittest.TestCase):
    """Checks the default config for scikit-learn selector classes."""

    def setUp(self):
        self.df = test_df().apply(close, axis=1)

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


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestAgumentors(unittest.TestCase):
    """Checks the default config for scikit-learn augmenting transformer classes."""

    def setUp(self):
        self.df = test_df().apply(close, axis=1)

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

    def test_RedoxAggregator(self):
        """Test the RedoxAggregator transfomer."""
        df = self.df
        tmr = RedoxAggregator()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_ElementAggregator(self):
        """Test the ElementAggregator transfomer."""
        df = self.df
        tmr = Devolatilizer()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)

    def test_LambdaTransformer(self):
        """Test the LambdaTransformer transfomer."""
        df = test_df(cols=REE()).apply(close, axis=1)
        tmr = LambdaTransformer()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)


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
