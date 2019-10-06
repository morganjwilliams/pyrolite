import unittest
from pyrolite.ext.alphamelts.util import default_data_dictionary
from collections import OrderedDict

class TestDefaultMELTSDict(unittest.TestCase):
    def test_default(self):
        D = default_data_dictionary()
        self.assertIsInstance(D, (OrderedDict, dict))
        self.assertIn("title", D)
        self.assertIn("initialize", D)
        self.assertIn("calculationMode", D)
        self.assertIn("constraints", D)
