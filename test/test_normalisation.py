import unittest
from pyrolite.normalisation import *


class TestScaleFunction(unittest.TestCase):
    """Tests scale function generator."""

    def test_same_units(self):
        """Checks exchange between values with the same units is unity."""
        pass

    def test_different_units(self):
        """Checks exchange between values with different units isn't unity."""
        pass

    def test_failure_on_unknown_units(self):
        """Checks the function raises when unknown units are used."""
        pass


class TestRefcomp(unittest.TestCase):
    """Tests reference composition model."""

    def test_construction(self):
        """Checks the model can build."""
        pass

    def test_aggregate_oxides(self):
        """Checks the model can aggregate oxide components."""
        pass

    def test_collect_vars(self):
        """Checks that the model can assemble a list of relevant variables."""
        pass

    def test_set_units(self):
        """Checks that the model can be represented as different units."""
        # Check function
        pass

    def test_set_units_reversible(self):
        """Checks that the unit conversion is reversible."""
        # Check reversible
        pass

    def test_normalize(self):
        """Checks that the model can be used for normalising a dataframe."""
        pass

class TestReferenceDB(unittest.TestCase):
    """Tests the formation of a reference dictionary from a directory."""


    def test_build(self):
        """Checks that the dictionary constructs."""
        pass

    def test_content(self):
        """Checks that the dictionary structure is correct."""
        pass


if __name__ == '__main__':
    unittest.main()
