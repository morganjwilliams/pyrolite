import unittest

from pyrolite.util.skl.helpers import get_PCA_component_labels
from pyrolite.util.synthetic import normal_frame

try:
    from sklearn.decomposition import PCA

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "Requires Scikit-learn")
class TestPCAComponentLabels(unittest.TestCase):
    """Checks the default config for scikit-learn imputing transformer classes."""

    def setUp(self):
        self.df = normal_frame()
        self.pca = PCA()

    def test_default(self):
        self.pca.fit(self.df)
        labels = get_PCA_component_labels(self.pca, self.df.columns)
        self.assertIsInstance(labels, list)

    @unittest.expectedFailure
    def test_pca_unfitted(self):
        labels = get_PCA_component_labels(self.pca, self.df.columns)


if __name__ == "__main__":
    unittest.main()
