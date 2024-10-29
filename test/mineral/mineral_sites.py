import unittest

from pyrolite.mineral.sites import AX, IX, MX, OX, TX, VX, Site


class TestSites(unittest.TestCase):
    """Test the Site base class and builtin sites."""

    def setUp(self):
        self.names = ["A", "B", "", ""]
        self.builtins = [MX, TX, IX, VX, OX, AX]

    def test_site_build(self):
        for S in [Site] + self.builtins:
            s = S()

    def test_site_eq(self):
        sites = [Site(name) for name in self.names]

    def test_repr(self):
        out = repr(Site())

    def test_str(self):
        out = str(Site())


if __name__ == "__main__":
    unittest.main()
