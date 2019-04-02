import unittest
import numpy as np
import periodictable as pt
from pyrolite.util.alphamelts.parse import *


class TestParseMELTSComposition(unittest.TestCase):
    def setUp(self):
        self.cstring = """Fe''0.18Mg0.83Fe'''0.04Al1.43Cr0.52Ti0.01O4"""

    def test_parse_dict(self):
        ret = from_melts_cstr(self.cstring, formula=False)
        self.assertTrue(isinstance(ret, dict))
        self.assertTrue("Fe{2+}" in ret.keys())
        self.assertTrue(np.isclose(ret["Fe{2+}"], 0.18))

    def test_parse_formula(self):
        ret = from_melts_cstr(self.cstring, formula=True)
        self.assertTrue(isinstance(ret, pt.formulas.Formula))


if __name__ == "__main__":
    unittest.main()
