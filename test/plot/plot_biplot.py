import logging
import unittest

from pyrolite.data.Aitchison import load_hongite
from pyrolite.plot.biplot import compositional_biplot

logger = logging.getLogger(__name__)


class TestCompositionalBiplot(unittest.TestCase):
    """Tests the compositional_biplot functionality."""

    def setUp(self):
        df = load_hongite()
        self.data = df.values
        self.labels = df.columns

    def test_default(self):
        a0 = compositional_biplot(self.data, labels=self.labels)
