import unittest
import numpy as np
from pyrolite.util.synthetic import normal_frame
from pyrolite.comp.codata import close

try:
    import sklearn
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, train_test_split
    import sklearn.manifold

    HAVE_SKLEARN = True

    def test_classifier():
        param_grid = dict(gamma=np.array([0.001, 0.01]), C=np.array([1, 10]))
        gs = GridSearchCV(SVC(gamma="scale"), param_grid, cv=2)
        return gs

    from pyrolite.util.skl.vis import *
    from pyrolite.util.skl.pipeline import SVC_pipeline

except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestPlotConfusionMatrix(unittest.TestCase):
    """Checks the plot_confusion_matrix matrix function."""

    def setUp(self):
        self.X = normal_frame(size=20).apply(close, axis=1)
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
        self.X = normal_frame(size=20).apply(close, axis=1)
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
class TestPlotMapping(unittest.TestCase):
    """Checks the plot_mapping function."""

    def setUp(self):
        self.X = normal_frame(size=20).apply(close, axis=1)
        iris = sklearn.datasets.load_iris()
        self.data, self.target = iris["data"], iris["target"]
        svc = SVC_pipeline(probability=True)
        self.gs = svc.fit(self.data, self.target)
        self.clf = self.gs.best_estimator_

    def test_default(self):
        a, tfm, mapped = plot_mapping(self.data, self.clf)

    def test_mapping_str(self):
        for mapping in ["MDS", "isomap"]:
            with self.subTest(mapping=mapping):
                a, tfm, mapped = plot_mapping(self.data, self.clf, mapping=mapping)

    def test_mapping_instance(self):
        for mapping in [
            sklearn.manifold.MDS(n_components=2, metric=True),
            sklearn.manifold.Isomap(n_components=2),
        ]:
            with self.subTest(mapping=mapping):
                a, tfm, mapped = plot_mapping(self.data, self.clf, mapping=mapping)

    def test_mapping_array(self):
        mapping = sklearn.manifold.MDS(n_components=2, metric=True).fit_transform(
            self.data
        )
        a, tfm, mapped = plot_mapping(self.data, self.clf, mapping=mapping)

    def test_Y_transformer_or_array(self):
        for Y in [self.clf, self.clf.predict(self.data)]:  # transformer, array
            with self.subTest(Y=Y):
                a, tfm, mapped = plot_mapping(self.data, Y)


if __name__ == "__main__":
    unittest.main()
