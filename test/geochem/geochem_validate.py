import unittest
from pyrolite.geochem.ind import REE
from pyrolite.geochem.validate import *


class TestIsChem(unittest.TestCase):
    """Checks the 'is a chem' function."""

    def setUp(self):
        self.ree = REE()

    def test_ischem_str(self):
        ret = ischem(self.ree[0])
        self.assertTrue(isinstance(ret, bool))
        self.assertTrue(ret)

    def test_notchem_str(self):
        ret = ischem("Notachemical")
        self.assertTrue(isinstance(ret, bool))
        self.assertFalse(ret)

    def test_ischem_list(self):
        ret = ischem(self.ree)
        self.assertTrue(isinstance(ret, list))
        self.assertTrue(all([isinstance(i, bool) for i in ret]))

is_isotoperatio

if __name__ == '__main__':
    unittest.main()
