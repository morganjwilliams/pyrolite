import unittest
import numpy as np
from pyrolite.util.synthetic import test_df
from pyrolite.comp.codata import close

try:
    import sklearn
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, train_test_split

    HAVE_SKLEARN = True

    def test_classifier():
        param_grid = dict(gamma=np.array([0.001, 0.01]), C=np.array([1, 10]))
        gs = GridSearchCV(SVC(), param_grid, cv=2)
        return gs

    from pyrolite.util.skl.vis import *

except ImportError:
    HAVE_SKLEARN = False



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

if __name__ == '__main__':
    unittest.main()
