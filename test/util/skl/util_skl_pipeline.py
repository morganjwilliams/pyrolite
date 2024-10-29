import unittest

import numpy as np

import pyrolite.comp
from pyrolite.util.general import remove_tempdir, temp_path
from pyrolite.util.synthetic import normal_frame

try:
    import sklearn
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    HAVE_SKLEARN = True

    def get_classifier():
        param_grid = dict(gamma=np.array([0.001, 0.01]), C=np.array([1, 10]))
        gs = GridSearchCV(SVC(gamma="scale"), param_grid, cv=2)
        return gs

    from pyrolite.util.skl.pipeline import *
    from pyrolite.util.skl.transform import LogTransform

except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestFitSaveClassifier(unittest.TestCase):
    def setUp(self):
        self.X, self.y = sklearn.datasets.load_iris(return_X_y=True)
        self.classifier = get_classifier()
        self.dir = temp_path()

    def test_default(self):
        clf = fit_save_classifier(self.classifier, self.X, self.y, directory=self.dir)

    def test_dataframe_X(self):
        clf = fit_save_classifier(
            self.classifier, pd.DataFrame(self.X), self.y, directory=self.dir
        )

    def tearDown(self):
        remove_tempdir(self.dir)


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestClassifierPerformanceReport(unittest.TestCase):
    def setUp(self):
        self.X, self.y = sklearn.datasets.load_iris(return_X_y=True)
        self.classifier = get_classifier()
        self.dir = temp_path()

    def test_default(self):
        self.classifier.fit(self.X, self.y)
        classifier_performance_report(
            self.classifier, self.X, self.y, directory=self.dir
        )

    def tearDown(self):
        remove_tempdir(self.dir)


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TesSVCPipleline(unittest.TestCase):
    def setUp(self):
        self.X, self.y = sklearn.datasets.load_iris(return_X_y=True)

    def test_default(self):
        clf = SVC_pipeline()
        clf.fit(self.X, self.y)
        self.assertIsInstance(clf, sklearn.model_selection.GridSearchCV)

    def test_scaler(self):
        clf = SVC_pipeline(scaler=sklearn.preprocessing.StandardScaler())
        clf.fit(self.X, self.y)
        self.assertIsInstance(clf, sklearn.model_selection.GridSearchCV)

    def test_pre_transform(self):
        pass
        # clf = SVC_pipeline(transform=LogTransform())
        # clf.fit(self.X, self.y)
        # self.assertIsInstance(clf, sklearn.model_selection.GridSearchCV)


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestPdUnion(unittest.TestCase):
    """Checks the default config for scikit-learn augmenting transformer classes."""

    def setUp(self):
        self.df = normal_frame().pyrocomp.renormalise()

    def test_PdUnion(self):
        """Test the PdUnion pipeline augmentor."""
        df = self.df
        tmr = PdUnion()
        for input in [df]:
            with self.subTest(input=input):
                out = tmr.transform(input)


if __name__ == "__main__":
    unittest.main()
