import logging
import unittest

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyrolite.data.Aitchison import *
from pyrolite.plot.biplot import *

logger = logging.getLogger(__name__)


class TestCompositionalBiplot(unittest.TestCase):
    """Tests the compositional_biplot functionality."""

    def setUp(self):
        df = load_hongite()
        self.data = df.values
        self.labels = df.columns

    def test_default(self):
        a0 = compositional_biplot(self.data, labels=self.labels)
