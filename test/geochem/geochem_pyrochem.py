import unittest
import numpy as np
import pyrolite.geochem
from pyrolite.comp.codata import renormalise
from pyrolite.util.synthetic import test_df, test_ser

# [print("# " + i) for i in dir(df.pyrochem) if "__" not in i and not i.startswith("_")]


class TestPyrochem(unittest.TestCase):
    """
    Test the pyrochem dataframe accessor interface to
    pyrolite.geochem functions.
    """

    def setUp(self):
        self.df = renormalise(test_df(index_length=4))

    def test_pyrochem_check_multiple_cation_inclusion(self):
        obj = self.df.copy()
        cations = obj.pyrochem.check_multiple_cation_inclusion()
        self.assertTrue(len(cations) == 0)

    # pyrolite.geochem.transform functions

    def test_pyrochem_add_MgNo(self):
        obj = self.df.copy()
        obj.pyrochem.add_MgNo()

    def test_pyrochem_add_MgNo_ferricferrous(self):
        obj = self.df.copy()
        obj.pyrochem.add_MgNo().pyrochem.recalculate_Fe(
            to=dict(FeO=0.9, Fe2O3=0.1)
        ).pyrochem.add_MgNo(name="Mg#2")

    def test_pyrochem_add_ratio(self):
        obj = self.df.copy()
        pass

    def test_pyrochem_aggregate_element(self):
        obj = self.df.copy()
        pass

    def test_pyrochem_devolatilise(self):
        obj = self.df.copy()
        pass

    def test_pyrochem_elemental_sum(self):
        obj = self.df.copy()
        pass

    def test_pyrochem_lambda_lnREE(self):
        obj = self.df.copy()
        pass

    def test_pyrochem_normalize_to(self):
        obj = self.df.copy()
        pass

    def test_pyrochem_recalculate_Fe(self):
        obj = self.df.copy()
        pass

    def test_pyrochem_to_molecular(self):
        obj = self.df.copy()
        pass

    def test_pyrochem_to_weight(self):
        obj = self.df.copy()
        pass
