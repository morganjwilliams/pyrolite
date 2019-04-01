import unittest
from pyrolite.util.general import internet_connection
from pyrolite.util.melts.util import default_data_dictionary
from pyrolite.util.melts.web import *

@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestWebService(unittest.TestCase):
    """Tests the current MELTS webservice interactivity with default data."""

    def setUp(self):
        self.dict = default_data_dictionary()

    def test_melts_compute(self):
        """Tests the MELTS-compute web service."""
        result = melts_compute(self.dict)

    def test_melts_oxides(self):
        """Tests the MELTS-oxides web service."""
        result = melts_oxides(self.dict)

    def test_melts_phases(self):
        """Tests the MELTS-phases web service."""
        result = melts_phases(self.dict)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
