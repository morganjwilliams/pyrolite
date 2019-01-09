import unittest
from pyrolite.mineral.sites import *


class TestSites(unittest.TestCase):
    """Test the Site base class and builtin sites."""

    def setUp(self):
        self.names = ["A", "B", "", ""]
        self.builtins = [MX, TX, IX, VX, OX, AX]

    def test_site_build(self):
        site = Site(self.names[0])

    def test_site_eq(self):
        sites = [Site(name) for name in self.names]

    def test_repr(self):
        pass

    def test_str(self):
        pass


if __name__ == "__main__":
    unittest.main()
