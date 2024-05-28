import unittest

from pyrolite.geochem.isotope.count import deadtime_correction


class TestDeadtimeCorrection(unittest.TestCase):
    def test_default(self):
        deadtime_correction(10000, 20)
